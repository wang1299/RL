import torch
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict
from components.perception.hm3d_labels import HM3D_LABEL_ALIASES

class GroundingDINODetector:
    def __init__(
        self,
        config_path,
        checkpoint_path,
        text_prompt,
        box_threshold=0.35,
        text_threshold=0.25,
        device=None,
        excluded_labels=None,
        max_box_area_ratio=1.0,
        max_box_aspect_ratio=100.0,
    ):
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Perception] Loading Grounding DINO on {self.device}...")
        
        # 加载模型
        try:
            self.model = load_model(config_path, checkpoint_path)
            self.model = self.model.to(self.device)
        except Exception as e:
            print(f"[Error] Failed to load Grounding DINO model: {e}")
            raise e

        self.text_prompts = self._split_text_prompt(text_prompt)
        self.text_prompt = self.text_prompts[0] if self.text_prompts else text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.excluded_labels = {str(label) for label in (excluded_labels or [])}
        self.max_box_area_ratio = float(max_box_area_ratio)
        if self.max_box_area_ratio <= 0:
            self.max_box_area_ratio = 1.0
        self.max_box_aspect_ratio = float(max_box_aspect_ratio)
        if self.max_box_aspect_ratio <= 0:
            self.max_box_aspect_ratio = 100.0
        print(
            f"[Perception] Grounding DINO prompt chunks: "
            f"{len(self.text_prompts)} ({sum(prompt.count('.') for prompt in self.text_prompts)} labels)"
        )

        # === 相机内参设置 (Camera Intrinsics) ===
        # 警告：必须根据你 config/env.json 或生成数据集时的设置修改这些值！
        # AI2-THOR/Habitat 默认通常是:
        self.hfov = 90.0       # 水平视场角
        self._set_image_geometry(300, 300)

    @torch.no_grad()
    def detect(self, rgb_image, depth_image=None, agent_state=None):
        """
        Args:
            rgb_image: (H, W, 3) numpy array
            depth_image: (H, W) or (H, W, 1) numpy array, 深度值(米)
            agent_state: dict, {'position': {'x':.., 'y':.., 'z':..}, 'rotation': {'x':.., 'y':.., 'z':.., 'w':..}}
        """
        H, W = rgb_image.shape[:2]
        self._set_image_geometry(W, H)

        # 1. 图像预处理
        image_pil = Image.fromarray(rgb_image.astype(np.uint8))
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image_pil, None)

        # 2. 模型推理。GroundingDINO/BERT has a limited text length, so large
        # HM3D vocabularies are split into several shorter prompts.
        prediction_batches = []
        for text_prompt in self.text_prompts:
            boxes, logits, phrases = self._predict_prompt(image_tensor, text_prompt)
            prediction_batches.append((boxes, logits, phrases))
        
        # OOM Fix: Clean up
        image_tensor = image_tensor.cpu()
        # torch.cuda.empty_cache() # Removed for speed

        # === [DEBUG] 打印检测结果 ===
        if any(len(phrases) > 0 for _, _, phrases in prediction_batches):
            # 防止刷屏太快，只打印前几个
            # print(f"[DEBUG] DINO detected: {phrases} | Scores: {[round(s.item(), 2) for s in logits]}")
            pass
        else:
            # 只有在非常确信需要看空结果时才打印，否则会刷屏
            # print("[DEBUG] DINO detected NOTHING")
            pass

        detections = []

        # 3. 处理检测结果
        for boxes, logits, phrases in prediction_batches:
            for box, score, label in zip(boxes, logits, phrases):
                # 还原 bbox 到像素坐标
                box = box * torch.Tensor([W, H, W, H])
                cx, cy, w, h = box.numpy()
                
                # 这里的 bbox 格式是 [min_x, min_y, max_x, max_y]
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                canonical_label = self._canonical_label(label)
                if self._should_skip_detection(canonical_label, bbox, W, H):
                    continue

                # === 核心：2D -> 3D 投影 ===
                world_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
                if depth_image is not None and agent_state is not None:
                    world_pos = self._project_to_3d(cx, cy, depth_image, agent_state)
                
                # 构造符合 LocalGraphBuilder 接口的数据
                detections.append({
                    "label": label,
                    "canonical_label": canonical_label,
                    "score": float(score),
                    "bbox": bbox,
                    "object_id": f"{label}_{np.random.randint(10000)}", # 临时ID
                    "position": world_pos
                })
            
        return detections

    def _set_image_geometry(self, width, height):
        self.img_width = max(int(width), 1)
        self.img_height = max(int(height), 1)
        self.fx = (self.img_width / 2.0) / np.tan(np.deg2rad(self.hfov / 2.0))
        self.fy = (self.img_height / 2.0) / np.tan(np.deg2rad(self.hfov / 2.0))
        self.cx = self.img_width / 2.0
        self.cy = self.img_height / 2.0

    @staticmethod
    def _canonical_label(label):
        key = " ".join(str(label or "").replace("_", " ").split()).lower()
        return HM3D_LABEL_ALIASES.get(key, key)

    def _should_skip_detection(self, canonical_label, bbox, image_width, image_height):
        if canonical_label in self.excluded_labels:
            return True

        if bbox is None or len(bbox) != 4:
            return False

        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1 = min(max(x1, 0.0), float(image_width))
        x2 = min(max(x2, 0.0), float(image_width))
        y1 = min(max(y1, 0.0), float(image_height))
        y2 = min(max(y2, 0.0), float(image_height))
        box_w = max(x2 - x1, 0.0)
        box_h = max(y2 - y1, 0.0)
        if box_w <= 0.0 or box_h <= 0.0:
            return True

        image_area = max(float(image_width * image_height), 1.0)
        area_ratio = (box_w * box_h) / image_area
        if self.max_box_area_ratio < 1.0 and area_ratio > self.max_box_area_ratio:
            return True

        aspect_ratio = max(box_w / max(box_h, 1e-6), box_h / max(box_w, 1e-6))
        if self.max_box_aspect_ratio < 100.0 and aspect_ratio > self.max_box_aspect_ratio:
            return True

        return False

    @staticmethod
    def _split_text_prompt(text_prompt, max_words=120):
        labels = [
            part.strip()
            for part in str(text_prompt or "").split(".")
            if part.strip()
        ]
        if not labels:
            return [str(text_prompt or "object").strip() + " ."]

        chunks = []
        current = []
        current_words = 0
        for label in labels:
            word_count = max(len(label.replace("/", " ").split()), 1)
            if current and current_words + word_count > max_words:
                chunks.append(" . ".join(current) + " .")
                current = []
                current_words = 0
            current.append(label)
            current_words += word_count
        if current:
            chunks.append(" . ".join(current) + " .")
        return chunks

    def _predict_prompt(self, image_tensor, text_prompt):
        try:
            return predict(
                model=self.model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
                remove_combined=True,
            )
        except (IndexError, RuntimeError) as exc:
            if "out of memory" in str(exc).lower():
                raise
            return predict(
                model=self.model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )

    def _project_to_3d(self, u, v, depth_img, agent_state):
        # 1. 读取深度
        u_int, v_int = int(u), int(v)
        v_int = max(0, min(v_int, self.img_height - 1))
        u_int = max(0, min(u_int, self.img_width - 1))
        
        d = depth_img[v_int, u_int]
        if isinstance(d, np.ndarray): d = d.item()
        
        # 简单过滤无效深度
        if d <= 0.01 or d > 10.0:
            return {'x': 0.0, 'y': 0.0, 'z': 0.0}

        # 2. 图像坐标 -> 相机坐标 (Camera Space)
        # 假设 Habitat 相机: +X 右, +Y 上, -Z 前 (或者 Z 是深度)
        # 根据经验通常 Z 是深度方向
        z_c = d  # 有些坐标系是 -d，需要根据你的具体环境测试
        x_c = (u - self.cx) * d / self.fx
        y_c = -(v - self.cy) * d / self.fy # 图像Y向下，世界Y向上，通常取反

        camera_point = np.array([x_c, y_c, z_c])

        # 3. 相机坐标 -> 世界坐标 (World Space)
        # AI2-THOR Agent Position 通常在地面 (y=0 附近)，但相机有高度
        # 我们假设 agent_pos 是相机位置，如果不是，需要加 offset (但这取决于 upstream 传什么)
        # 通常 event.metadata['agent']['position'] 是 floor level
        # event.metadata['agent']['cameraHorizon'] 是俯仰角
        
        agent_pos = np.array([
            agent_state['position']['x'],
            agent_state['position']['y'],
            agent_state['position']['z']
        ])
        
        # 处理旋转 (支持 Quaternion 或 Euler)
        rot = agent_state['rotation']
        r = None
        
        if isinstance(rot, dict):
            if 'w' in rot:
                # Quaternion [x, y, z, w]
                r = R.from_quat([rot['x'], rot['y'], rot['z'], rot['w']])
            else:
                # AI2-THOR default: Euler degrees {x, y, z}. Usually only Y (yaw) changes for nav.
                # standard order for 'rotation' dict in ai2thor is likely (x, y, z) but usually y is yaw.
                # Assuming 'y' is yaw (rotation around vertical axis). 
                # 注意：AI2-THOR 坐标系转换可能需要更精细处理，这里做基础兼容
                r = R.from_euler('y', rot['y'], degrees=True)
        else:
            # 假设是 list/array
            if len(rot) == 4:
                r = R.from_quat(rot)
            else:
                 # 假设 Euler [x, y, z]
                r = R.from_euler('xyz', rot, degrees=True)

        if r is None:
             return {'x': 0.0, 'y': 0.0, 'z': 0.0}

        # 考虑相机俯仰角 (Camera Horizon / Pitch)
        # AI2-THOR: positive horizon looks down? No, usually positive is down/up?
        # Check docs: "cameraHorizon": 0 is straight. 30 is looking down 30 degrees?
        # 简单起见，如果不为0，叠加一个 x 轴旋转
        if 'cameraHorizon' in agent_state and abs(agent_state['cameraHorizon']) > 1e-3:
             r_pitch = R.from_euler('x', agent_state['cameraHorizon'], degrees=True)
             r = r * r_pitch

        # 应用旋转
        # 这里的相乘顺序取决于他是 局部旋转 还是 全局旋转
        # world_vec = R_agent * vec_camera
        world_point_relative = r.apply(camera_point)
        
        # 加上 Agent 位置 (注意：这里忽略了相机相对于 Agent 的高度偏移，如果 agent_pos 是脚底板)
        # 如果需要更精确，应该加 camera height，通常是 0.675 或类似
        # 临时修复：如果是 'y' ~ 0.9 这种，可能是脚底。
        # 但 depth 投影出来的 y_c 在相机系下。
        world_point = world_point_relative + agent_pos

        return {'x': world_point[0], 'y': world_point[1], 'z': world_point[2]}
