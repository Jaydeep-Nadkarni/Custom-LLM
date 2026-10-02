import torch
from torch.utils.data import DataLoader

from data.dataset import TokenDataset


dataset = TokenDataset(
    "data/tokenized/fineweb_train_inputs.npy",
    "data/tokenized/fineweb_train_targets.npy",
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)

inputs, targets = next(iter(dataloader))

print("Dataset size :", len(dataset))
print("Input batch  :", inputs.shape)
print("Target batch :", targets.shape)
print("Input dtype  :", inputs.dtype)
print("Target dtype :", targets.dtype)
