import torch

from model.attention import CausalSelfAttention


torch.manual_seed(42)
attention = CausalSelfAttention(
    d_model=384,
    n_heads=6,
)
attention.eval()

# Create a sequence
x = torch.randn(1, 8, 384)

# Original output
output1 = attention(x)

# Change ONLY the last token
x_modified = x.clone()
x_modified[:, -1, :] = torch.randn(384)

# New output
output2 = attention(x_modified)

# Compare earlier positions
difference = torch.abs(
    output1[:, :-1] - output2[:, :-1]
).max().item()

print("Maximum difference in previous positions:", difference)
print("Causal attention passed:", difference == 0.0)
