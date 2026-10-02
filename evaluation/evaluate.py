import argparse
from pathlib import Path
import torch
import yaml
from model import DecoderLM
from evaluation.perplexity import perplexity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/10m.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pt"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = DecoderLM(**config["model"])
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print("Evaluation scaffold loaded. Add a validation dataloader to compute loss and perplexity.")
    print(f"Example metric conversion: perplexity(torch.tensor(2.0)) = {perplexity(torch.tensor(2.0)):.2f}")


if __name__ == "__main__":
    main()
