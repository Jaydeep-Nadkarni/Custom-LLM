import torch

from model.rmsnorm import RMSNorm


hidden_size = 384
batch_size = 2
sequence_length = 512

# Create random input
x = torch.randn(
    batch_size,
    sequence_length,
    hidden_size,
)

# Create RMSNorm
norm = RMSNorm(hidden_size)

# Forward pass
output = norm(x)

# Results
print("Input shape :", x.shape)
print("Output shape:", output.shape)
print(
    "Parameter count:",
    sum(parameter.numel() for parameter in norm.parameters()),
)
print("Output mean:", output.mean().item())
print("Output RMS:", torch.sqrt(torch.mean(output ** 2)).item())
