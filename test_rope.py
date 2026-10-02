import torch

from model.rope import RotaryEmbedding


head_dim = 64
sequence_length = 512
batch_size = 2
num_heads = 6

# Create RoPE
rope = RotaryEmbedding(
    head_dim=head_dim,
    max_seq_len=sequence_length,
)

# Dummy query tensor
q = torch.randn(
    batch_size,
    num_heads,
    sequence_length,
    head_dim,
)

# Apply RoPE rotation
output = rope(q, sequence_length)

# Print results
print("Q shape   :", q.shape)
print("Output shape:", output.shape)
print("Output dtype:", output.dtype)
