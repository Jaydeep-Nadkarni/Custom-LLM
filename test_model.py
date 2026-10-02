import torch

from model.model import TransformerLM


model = TransformerLM(
    vocab_size=32000,
    d_model=384,
    n_layers=6,
    n_heads=6,
    ffn_hidden_size=1024,
    max_seq_len=512,
)

# Count parameters
total_params = sum(
    parameter.numel()
    for parameter in model.parameters()
)

# Count trainable parameters
trainable_params = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Total parameters    :", total_params)
print("Trainable parameters:", trainable_params)

# Test forward pass
input_ids = torch.randint(
    0,
    32000,
    (2, 512),
)

with torch.no_grad():
    logits = model(input_ids)

print("Input shape :", input_ids.shape)
print("Logits shape:", logits.shape)
