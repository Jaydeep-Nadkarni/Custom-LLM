import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=512, base=10000):
        super().__init__()

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2).float() / head_dim
            )
        )

        positions = torch.arange(max_seq_len).float()

        freqs = torch.outer(positions, inv_freq)

        self.register_buffer("cos_cached", torch.cos(freqs))
        self.register_buffer("sin_cached", torch.sin(freqs))

    def forward(self, x, seq_len):
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        return cos, sin


def apply_rope(x, theta=10000.0):
    """Apply rotary position embeddings to [batch, heads, sequence, head_dim]."""
    _, _, sequence_length, head_dim = x.shape
    inverse_frequency = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype) / head_dim)
    )
    positions = torch.arange(sequence_length, device=x.device, dtype=x.dtype)
    angles = torch.outer(positions, inverse_frequency)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)