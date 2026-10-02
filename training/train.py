"""Minimal training entry point. Replace the random batches with tokenized data."""
import argparse
from pathlib import Path
import torch
import yaml
from model import DecoderLM
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler
from training.checkpoint import save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/10m.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DecoderLM(**config["model"]).to(device)
    optimizer = build_optimizer(model, config["training"]["learning_rate"], config["training"]["weight_decay"])
    scheduler = build_scheduler(optimizer, config["training"]["warmup_steps"], config["training"]["steps"])
    model.train()
    for step in range(config["training"]["steps"]):
        tokens = torch.randint(config["model"]["vocab_size"], (config["training"]["batch_size"], config["model"]["context_length"] + 1), device=device)
        _, loss = model(tokens[:, :-1], tokens[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step % config["training"]["log_every"] == 0:
            print(f"step={step} loss={loss.item():.4f}")
    save_checkpoint(Path("checkpoints/latest.pt"), model, optimizer, scheduler, config["training"]["steps"], config["model"])


if __name__ == "__main__":
    main()
