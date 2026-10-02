import math
import torch


def perplexity(loss: torch.Tensor) -> float:
    return math.exp(min(float(loss.detach()), 20.0))
