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
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        first = x[..., ::2]
        second = x[..., 1::2]
        rotated = torch.stack(
            [
                first * cos - second * sin,
                first * sin + second * cos,
            ],
            dim=-1,
        )
        return rotated.flatten(-2)