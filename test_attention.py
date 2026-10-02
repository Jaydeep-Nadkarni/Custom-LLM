import torch

from model.attention import CausalSelfAttention


# Model configuration
batch_size = 2
sequence_length = 512
d_model = 384
n_heads = 6

# Create random input
x = torch.randn(
    batch_size,
    sequence_length,
    d_model,
)

# Create attention layer
attention = CausalSelfAttention(
    d_model=d_model,
    n_heads=n_heads,
)

# Forward pass
output = attention(x)

# Results
print("Input shape :", x.shape)
print("Output shape:", output.shape)
print(
    "Parameter count:",
    sum(parameter.numel() for parameter in attention.parameters()),
)
