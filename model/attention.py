import torch
from torch import nn
from torch.nn import functional as F
from .rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = RotaryEmbedding(self.head_dim)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, d_model = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, length, self.n_heads, self.head_dim)
        query = self.rope(query.view(shape).transpose(1, 2), length)
        key = self.rope(key.view(shape).transpose(1, 2), length)
        value = value.view(shape).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.output(attended.transpose(1, 2).reshape(batch, length, d_model))
