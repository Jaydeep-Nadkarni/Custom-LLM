import torch


def build_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float = 0.1) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
