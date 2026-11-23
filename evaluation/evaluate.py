"""Full (non-sampled) validation of a Custom-LLM V3 checkpoint.

Uses the dataset, evaluation and checkpoint-format definitions from
training/train_v3.py so results match the losses logged during training.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inference.generate import build_model_from_config
from training.train_v3 import CHECKPOINT_FORMAT, V3TokenDataset, evaluate


def validation_datasets(config: dict) -> dict:
    """Same validation shards that train_v3.py evaluates on."""
    root = Path(config["data"]["val_inputs"]).parent
    seq_len = config["model"]["context_length"]
    return {
        name: V3TokenDataset(
            root / f"inputs/{name}_validation_00000.npy",
            root / f"targets/{name}_validation_00000.npy",
            seq_len,
        )
        for name in ("fineweb", "wikipedia")
    }


def run_full_validation(model, config, device, batch_size=8, num_workers=0, use_amp=None) -> dict:
    device = torch.device(device)
    if use_amp is None:
        use_amp = device.type == "cuda"
    vocab_size = config["model"]["vocab_size"]
    results = {}
    sizes = {}
    for name, dataset in validation_datasets(config).items():
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        loss = evaluate(model, loader, device, vocab_size, use_amp)
        results[f"{name}_loss"] = loss
        results[f"{name}_ppl"] = math.exp(min(loss, 20))
        results[f"{name}_sequences"] = len(dataset)
        sizes[name] = len(dataset)
    combined = sum(results[f"{n}_loss"] * s for n, s in sizes.items()) / sum(sizes.values())
    results["combined_weighted_loss"] = combined
    results["combined_ppl"] = math.exp(min(combined, 20))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/100m_v3.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON results file")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Not a {CHECKPOINT_FORMAT} checkpoint: {args.checkpoint}")
    if state["config"]["model"] != config["model"]:
        raise ValueError("Checkpoint architecture does not match the config")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config["model"])
    model.load_state_dict(state["model"])
    model.to(device)

    results = run_full_validation(
        model, config, device, args.batch_size, args.num_workers,
        use_amp=(device.type == "cuda" and not args.no_amp),
    )
    results["checkpoint"] = str(args.checkpoint)
    results["step"] = int(state["step"])
    print(json.dumps(results, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
