import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(hidden_size)
        )

    def forward(self, x):
        rms = torch.sqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps
        )

        return (x / rms) * self.weight