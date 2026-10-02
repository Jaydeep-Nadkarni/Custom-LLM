import torch


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return max(1, step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + (1.0 - progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
