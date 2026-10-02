import torch
from torch import nn
from .embeddings import TokenEmbedding
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock


class DecoderLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, hidden_dim: int, context_length: int = 2048, dropout: float = 0.0) -> None:
        super().__init__()
        self.context_length = context_length
        self.embeddings = TokenEmbedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, hidden_dim, dropout) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.embedding.weight

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        hidden = self.embeddings(token_ids)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.norm(hidden))
        loss = None if targets is None else nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
