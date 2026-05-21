import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from components.utils.observation import Observation


def list_episode_files(data_dir):
    return sorted(Path(data_dir).expanduser().resolve().glob("**/*.npz"))


class HM3DImitationLearningDataset(Dataset):
    """Sequence-window dataset for Habitat/HM3D expert trajectories."""

    def __init__(self, data_dir, seq_len=16):
        super().__init__()
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.seq_len = int(seq_len)
        self.files = list_episode_files(self.data_dir)
        if not self.files:
            raise FileNotFoundError(f"No HM3D IL .npz episodes found in {self.data_dir}")

        self.windows = []
        self.num_actions = None
        for path in self.files:
            with np.load(path, allow_pickle=False) as data:
                actions = data["actions"]
                num_steps = int(len(actions))
                file_num_actions = int(data["num_actions"][0]) if "num_actions" in data else 3
            if self.num_actions is None:
                self.num_actions = file_num_actions
            elif self.num_actions != file_num_actions:
                raise ValueError(
                    f"Inconsistent num_actions in {path}: {file_num_actions} != {self.num_actions}"
                )
            for start in range(0, num_steps, self.seq_len):
                length = min(self.seq_len, num_steps - start)
                if length > 0:
                    self.windows.append({"path": path, "start_idx": start, "length": length})

        meta_path = self.data_dir / "metadata.json"
        self.metadata = {}
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        entry = self.windows[idx]
        start = int(entry["start_idx"])
        end = start + int(entry["length"])
        with np.load(entry["path"], allow_pickle=False) as data:
            rgbs = data["rgb"][start:end]
            actions = data["actions"][start:end].astype(np.int64)
            last_actions = data["last_actions"][start:end].astype(np.int64)
            agent_pos = data["agent_pos"][start:end].astype(np.float32)
            scores = data["score"][start:end].astype(np.float32)
            coverages = data["coverage"][start:end].astype(np.float32)

        obs = []
        for rgb, pos, score, coverage in zip(rgbs, agent_pos, scores, coverages):
            obs.append(
                Observation(
                    [rgb, None, None, None],
                    info={
                        "agent_pos": (float(pos[0]), float(pos[1])),
                        "score": float(score),
                        "coverage": float(coverage),
                    },
                )
            )
        tgt_act = [torch.tensor(int(action), dtype=torch.long) for action in actions]
        return obs, last_actions.tolist(), tgt_act, len(obs)
