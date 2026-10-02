import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )

    def forward(self, input_ids):
        return self.embedding(input_ids)