import argparse
from pathlib import Path
import sys

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import TokenDataset
from model.model import TransformerLM
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler


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

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)


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
    # Training configuration
    # -----------------------------

    training_config = config["training"]


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
        training_config["steps"]
    )


    # -----------------------------
    # Initialization test
    # -----------------------------

    model.train()

    batch_size = training_config["batch_size"]
    context_length = model_config["context_length"]
    vocab_size = model_config["vocab_size"]

    test_input = torch.randint(
        0,
        vocab_size,
        (batch_size, context_length),
        device=device
    )

    with torch.no_grad():
        logits = model(test_input)

    print("Test input shape :", test_input.shape)
    print("Test logits shape:", logits.shape)

    print("Training setup initialized successfully.")


if __name__ == "__main__":
    main()