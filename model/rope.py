import torch


def apply_rope(x: torch.Tensor, theta: float = 10_000.0) -> torch.Tensor:
    """Apply rotary position embeddings to [batch, heads, length, head_dim]."""
    _, _, length, head_dim = x.shape
    positions = torch.arange(length, device=x.device, dtype=x.dtype)
    frequencies = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype) / head_dim))
    angles = positions[:, None] * frequencies[None, :]
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    first, second = x[..., ::2], x[..., 1::2]
    rotated = torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1)
    return rotated.flatten(-2)
