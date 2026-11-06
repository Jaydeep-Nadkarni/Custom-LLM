import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class TokenDataset(Dataset):
    def __init__(self, input_path, target_path):
        self.inputs = self._load_shards(input_path)
        self.targets = self._load_shards(target_path)
        self.lengths = [len(array) for array in self.inputs]
        if self.lengths != [len(array) for array in self.targets]:
            raise ValueError("Inputs and targets must have the same length")

    @staticmethod
    def _load_shards(path):
        path = Path(path)
        paths = sorted(path.glob("*.npy")) if path.is_dir() else [path]
        if not paths:
            raise FileNotFoundError(f"No NumPy shards found at {path}")
        return [np.load(shard, mmap_mode="r") for shard in paths]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        for inputs, targets, length in zip(self.inputs, self.targets, self.lengths):
            if index < length:
                input_ids = torch.from_numpy(inputs[index].astype(np.int64))
                target_ids = torch.from_numpy(targets[index].astype(np.int64))
                return input_ids, target_ids
            index -= length
        raise IndexError("Dataset index out of range")
