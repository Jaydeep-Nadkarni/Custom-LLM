import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import TokenDataset
from model.model import TransformerLM


# -----------------------------
# Configuration
# -----------------------------

vocab_size = 32000
d_model = 384
n_layers = 6
n_heads = 6
ffn_hidden_size = 1024
context_length = 512
batch_size = 4


# -----------------------------
# Dataset
# -----------------------------

dataset = TokenDataset(
    "data/tokenized/fineweb_train_inputs.npy",
    "data/tokenized/fineweb_train_targets.npy"
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)


# -----------------------------
# Model
# -----------------------------

model = TransformerLM(
    vocab_size=vocab_size,
    d_model=d_model,
    n_layers=n_layers,
    n_heads=n_heads,
    ffn_hidden_size=ffn_hidden_size,
    max_seq_len=context_length
)


# -----------------------------
# Get one batch
# -----------------------------

inputs, targets = next(iter(dataloader))


# -----------------------------
# Forward pass
# -----------------------------

logits = model(inputs)


# -----------------------------
# Calculate loss
# -----------------------------

loss_fn = nn.CrossEntropyLoss()

loss = loss_fn(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1)
)


print("Input shape :", inputs.shape)
print("Logits shape:", logits.shape)
print("Target shape:", targets.shape)
print("Loss        :", loss.item())