import torch


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[..., -1, None]] = float("-inf")
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
