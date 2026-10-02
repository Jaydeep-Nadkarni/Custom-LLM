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

embedding_weight = model.embedding.embedding.weight
print("Embedding weight mean :", embedding_weight.mean().item())
print("Embedding weight std  :", embedding_weight.std().item())
print("Embedding weight min  :", embedding_weight.min().item())
print("Embedding weight max  :", embedding_weight.max().item())


# -----------------------------
# Get one batch
# -----------------------------

inputs, targets = next(iter(dataloader))


# -----------------------------
# Embedding-only diagnostic
# -----------------------------

with torch.no_grad():
    x = model.embedding(inputs)
    x = model.norm(x)
    test_logits = torch.nn.functional.linear(
        x,
        model.embedding.embedding.weight,
    ) / (model.d_model ** 0.5)
    test_loss = nn.CrossEntropyLoss()(
        test_logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )

print("Embedding-only loss:", test_loss.item())


# -----------------------------
# Forward pass
# -----------------------------

logits = model(inputs)

print("Final logits std:", logits.std().item())

print("Logits mean :", logits.mean().item())
print("Logits std  :", logits.std().item())
print("Logits min  :", logits.min().item())
print("Logits max  :", logits.max().item())


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