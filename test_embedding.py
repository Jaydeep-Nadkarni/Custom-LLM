import torch

from model.embeddings import TokenEmbedding


# Model configuration
vocab_size = 32000
hidden_size = 384
batch_size = 2
sequence_length = 512

# Create random token IDs
input_ids = torch.randint(
    0,
    vocab_size,
    (batch_size, sequence_length),
)

# Create embedding layer
embedding = TokenEmbedding(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
)

# Forward pass
output = embedding(input_ids)

# Print results
print("Input shape :", input_ids.shape)
print("Output shape:", output.shape)
print("Parameter count:", sum(parameter.numel() for parameter in embedding.parameters()))

print("Embedding dimension:", output.shape[-1])