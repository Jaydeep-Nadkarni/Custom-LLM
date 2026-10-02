from torch import nn
from .attention import CausalSelfAttention
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.feed_forward_norm = RMSNorm(d_model)
        self.feed_forward = SwiGLU(d_model, hidden_dim)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))
