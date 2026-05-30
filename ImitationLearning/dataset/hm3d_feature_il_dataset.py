import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def list_feature_files(data_dir):
    return sorted(Path(data_dir).expanduser().resolve().glob("**/*.npz"))


class HM3DFeatureImitationLearningDataset(Dataset):
    """Sequence-window dataset backed by cached RGB encoder features."""

    returns_features = True

    def __init__(self, data_dir, seq_len=16, feature_key="rgb_features"):
        super().__init__()
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.seq_len = int(seq_len)
        self.feature_key = str(feature_key)
        self.files = list_feature_files(self.data_dir)
        if not self.files:
            raise FileNotFoundError(f"No cached HM3D IL .npz files found in {self.data_dir}")

        self.windows = []
        self.num_actions = None
        self.feature_dim = None
        for path in self.files:
            with np.load(path, allow_pickle=False) as data:
                if self.feature_key not in data:
                    raise KeyError(f"{path} does not contain {self.feature_key!r}")
                actions = data["actions"]
                features = data[self.feature_key]
                num_steps = int(len(actions))
                file_num_actions = int(data["num_actions"][0]) if "num_actions" in data else 3
                file_feature_dim = int(features.shape[-1])

            if self.num_actions is None:
                self.num_actions = file_num_actions
            elif self.num_actions != file_num_actions:
                raise ValueError(
                    f"Inconsistent num_actions in {path}: {file_num_actions} != {self.num_actions}"
                )

            if self.feature_dim is None:
                self.feature_dim = file_feature_dim
            elif self.feature_dim != file_feature_dim:
                raise ValueError(
                    f"Inconsistent feature dim in {path}: {file_feature_dim} != {self.feature_dim}"
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
            features = data[self.feature_key][start:end].astype(np.float32)
            actions = data["actions"][start:end].astype(np.int64)
            last_actions = data["last_actions"][start:end].astype(np.int64)

        tgt_act = torch.as_tensor(actions, dtype=torch.long)
        return features, last_actions.tolist(), tgt_act, int(len(actions))

    @staticmethod
    def seq_collate(batch):
        feature_list, last_list, tgt_list, lengths = zip(*batch)
        batch_size = len(batch)
        max_t = max(int(length) for length in lengths)
        feature_dim = int(feature_list[0].shape[-1])

        features = torch.zeros(batch_size, max_t, feature_dim, dtype=torch.float32)
        last_act = torch.full((batch_size, max_t), -100, dtype=torch.long)
        tgt_act = torch.full((batch_size, max_t), -100, dtype=torch.long)
        mask = []

        for idx, (feat, last, tgt, length) in enumerate(zip(feature_list, last_list, tgt_list, lengths)):
            length = int(length)
            features[idx, :length] = torch.as_tensor(feat, dtype=torch.float32)
            last_act[idx, :length] = torch.as_tensor(last[:length], dtype=torch.long)
            tgt_act[idx, :length] = tgt[:length].long()
            mask.append([1] * length + [0] * (max_t - length))

        x_batch = {
            "rgb_features": features,
            "lssg": None,
            "gssg": None,
            "lssg_mask": mask,
            "gssg_mask": mask,
            "agent_pos": None,
        }
        return x_batch, last_act, tgt_act, torch.as_tensor(lengths, dtype=torch.long)
