import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import TokenDataset
from model.model import TransformerLM
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler
from training.checkpoint import save_checkpoint


def main() -> None:
    # -----------------------------
    # Load configuration
    # -----------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/debug.yaml")
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-file", type=Path, default=Path("logs/training.log"))

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    training_config = config["training"]
    total_steps = args.steps or training_config["steps"]


    # -----------------------------
    # Device
    # -----------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)


    # -----------------------------
    # Model
    # -----------------------------

    model_config = config["model"]

    model = TransformerLM(
        vocab_size=model_config["vocab_size"],
        d_model=model_config["d_model"],
        n_layers=model_config["n_layers"],
        n_heads=model_config["n_heads"],
        ffn_hidden_size=model_config["hidden_dim"],
        max_seq_len=model_config["context_length"],
        dropout=model_config["dropout"],
    ).to(device)


    # -----------------------------
    # Parameter count
    # -----------------------------

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Model parameters:", total_params)


    # -----------------------------
    # Dataset
    # -----------------------------

    data_config = config["data"]

    dataset = TokenDataset(
        data_config["train_inputs"],
        data_config["train_targets"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True
    )

    print("Dataset size:", len(dataset))
    
    # -----------------------------
    # Optimizer
    # -----------------------------

    optimizer = build_optimizer(
        model,
        training_config["learning_rate"],
        training_config["weight_decay"]
    )


    # -----------------------------
    # Learning-rate scheduler
    # -----------------------------

    scheduler = build_scheduler(
        optimizer,
        training_config["warmup_steps"],
        total_steps
    )


    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    model.train()
    data_iterator = iter(dataloader)
    with args.log_file.open("a", encoding="utf-8") as log_file:
        for step in range(1, total_steps + 1):
            try:
                inputs, targets = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                inputs, targets = next(data_iterator)

            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, model_config["vocab_size"]),
                targets.reshape(-1),
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            learning_rate = optimizer.param_groups[0]["lr"]
            message = (
                f"step={step} loss={loss.item():.4f} "
                f"lr={learning_rate:.8f} grad_norm={float(gradient_norm):.4f}"
            )
            if step == 1 or step % training_config["log_every"] == 0 or step == total_steps:
                print(message)
                log_file.write(message + "\n")
                log_file.flush()

            if step % args.checkpoint_every == 0 or step == total_steps:
                save_checkpoint(
                    Path("checkpoints/latest.pt"),
                    model,
                    optimizer,
                    scheduler,
                    step,
                    config,
                )

    print(f"Training smoke run completed: {total_steps} steps")


if __name__ == "__main__":
    main()