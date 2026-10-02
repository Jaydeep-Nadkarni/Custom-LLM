import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(
            d_model,
            hidden_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            d_model,
            hidden_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            hidden_size,
            d_model,
            bias=False,
        )

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
