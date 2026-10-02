import torch
import torch.nn as nn
import torch.nn.functional as F
from .embeddings import TokenEmbedding
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_hidden_size: int,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_size=d_model,
        )
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ffn_hidden_size=ffn_hidden_size,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return F.linear(x, self.embedding.embedding.weight)
