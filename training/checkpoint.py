from pathlib import Path
import torch


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    return int(state["step"])
