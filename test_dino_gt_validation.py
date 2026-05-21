import numpy as np

from components.environments.habitat_env import HabitatEnv
from components.utils.observation import Observation
from RL_training.runner.parallel_habitat_rl_train_runner import ParallelHabitatRLTrainRunner


def _make_env():
    env = object.__new__(HabitatEnv)
    env.semantic_id_to_label = {1: "counter", 2: "sink", 3: "paper towel"}
    env.gt_validation_iou_threshold = 0.10
    env.gt_validation_mode = "relaxed"
    env.reward_excluded_labels = set()
    env.dino_max_box_area_ratio = 1.0
    env.dino_max_box_aspect_ratio = 100.0
    env._last_rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    env._last_semantic = np.zeros((20, 20), dtype=np.int32)
    env._last_semantic[1:10, 1:10] = 1
    env._last_semantic[10:19, 1:10] = 2
    env._last_semantic[5:15, 12:19] = 3
    env.current_ep_dir = None
    env.step_count = 0
    env.save_debug_interval = 100
    return env


def test_compound_dino_label_is_rejected():
    env = _make_env()
    dets = [{"label": "sink plant", "score": 0.9, "bbox": [1, 1, 10, 10]}]

    out = env.validate_detections(dets)

    assert out[0]["is_gt_valid"] is False
    assert out[0]["reject_reason"] in {"ambiguous_label", "compound_label"}


def test_relaxed_alias_can_match_semantic_gt():
    env = _make_env()
    dets = [{"label": "washbasin", "score": 0.9, "bbox": [1, 10, 10, 19]}]

    out = env.validate_detections(dets)

    assert out[0]["is_gt_valid"] is True
    assert out[0]["canonical_label"] == "Sink"
    assert out[0]["gt_semantic_id"] == 2


def test_low_iou_is_rejected():
    env = _make_env()
    dets = [{"label": "sink", "score": 0.9, "bbox": [0, 0, 2, 2]}]

    out = env.validate_detections(dets)

    assert out[0]["is_gt_valid"] is False
    assert out[0]["reject_reason"] == "low_iou"


def test_same_semantic_id_only_validates_once():
    env = _make_env()
    dets = [
        {"label": "counter", "score": 0.9, "bbox": [1, 1, 10, 10]},
        {"label": "counter", "score": 0.8, "bbox": [1, 1, 10, 10]},
    ]

    out = env.validate_detections(dets)

    assert sum(1 for det in out if det["is_gt_valid"]) == 1


def test_relaxed_iou_only_match_for_unknown_gt_vocab():
    env = _make_env()
    env.semantic_id_to_label[1] = "hm3d_custom_surface"
    dets = [{"label": "counter", "score": 0.9, "bbox": [1, 1, 10, 10]}]

    out = env.validate_detections(dets)

    assert out[0]["is_gt_valid"] is True
    assert out[0]["gt_match_mode"] == "semantic_iou_only"
    assert out[0]["gt_semantic_id"] == 1


def test_scene_reward_gt_ids_exclude_structural_labels():
    env = object.__new__(HabitatEnv)
    env.semantic_id_to_label = {1: "wall", 2: "floor", 3: "window", 4: "counter", 5: "sink"}
    env.reward_excluded_labels = {"Wall", "Floor", "Window"}

    reward_ids = env._build_scene_reward_gt_ids()

    assert reward_ids == {4, 5}


def test_runner_reward_filters_excluded_and_iou_only_detections():
    runner = object.__new__(ParallelHabitatRLTrainRunner)
    runner.detection_service = object()
    runner.det_score_thr = 0.20
    runner.reward_excluded_labels = {"Wall", "Floor", "Window"}
    runner.reward_allow_semantic_iou_only = False
    runner.discovery_bonus_scale = 5.0
    runner.per_env_discovered_objects = [set()]
    runner.per_env_discovered_instances = [set()]
    runner.per_env_discovered_gt_ids = [set()]
    runner.per_env_prev_score = [0.0]

    detections = [
        {"is_gt_valid": True, "score": 0.9, "canonical_label": "Wall", "gt_semantic_id": 1},
        {"is_gt_valid": True, "score": 0.9, "canonical_label": "Counter", "gt_semantic_id": 2, "gt_match_mode": "semantic_iou_only"},
        {"is_gt_valid": True, "score": 0.9, "canonical_label": "Sink", "gt_semantic_id": 3},
    ]
    obs = Observation([None, None, None, None], 0.0, False, False, {"scene_reward_gt_ids": [2, 3]})

    score, bonus = runner._apply_detection_reward(0, obs, detections)

    assert score == 0.5
    assert bonus == 2.5
    assert runner.per_env_discovered_gt_ids[0] == {3}
