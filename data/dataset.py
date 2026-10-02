import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, input_path, target_path):
        self.inputs = np.load(input_path, mmap_mode="r")
        self.targets = np.load(target_path, mmap_mode="r")
        if len(self.inputs) != len(self.targets):
            raise ValueError("Inputs and targets must have the same length")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_ids = torch.from_numpy(
            self.inputs[index].astype(np.int64)
        )
        target_ids = torch.from_numpy(
            self.targets[index].astype(np.int64)
        )
        return input_ids, target_ids
