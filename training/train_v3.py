
import argparse
import bisect
import json
import random
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from torch.utils.data import Dataset, DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.model import TransformerLM
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler


# ------------------------------------------------------------
# Dataset with efficient multi-shard indexing
# ------------------------------------------------------------

class V3TokenDataset(Dataset):

    def __init__(self, input_path, target_path, expected_seq_len):

        self.input_path = Path(input_path)
        self.target_path = Path(target_path)
        self.expected_seq_len = expected_seq_len

        self.input_files = self._get_files(self.input_path)
        self.target_files = self._get_files(self.target_path)

        input_names = [p.name for p in self.input_files]
        target_names = [p.name for p in self.target_files]

        if input_names != target_names:
            raise ValueError(
                "Input and target shard filenames do not match.\n"
                f"Inputs: {input_names}\nTargets: {target_names}"
            )

        self.inputs = []
        self.targets = []
        self.lengths = []
        self.cumulative_ends = []

        total = 0

        for input_file, target_file in zip(
            self.input_files, self.target_files
        ):

            x = np.load(input_file, mmap_mode="r")
            y = np.load(target_file, mmap_mode="r")

            if x.shape != y.shape:
                raise ValueError(
                    f"Shape mismatch: {input_file.name}"
                )

            if x.ndim != 2 or x.shape[1] != expected_seq_len:
                raise ValueError(
                    f"Unexpected shape {x.shape} in {input_file.name}"
                )

            self.inputs.append(x)
            self.targets.append(y)

            length = len(x)
            self.lengths.append(length)

            total += length
            self.cumulative_ends.append(total)

        self.total_sequences = total

        if total == 0:
            raise ValueError("Dataset contains no sequences")

    @staticmethod
    def _get_files(path):

        if path.is_dir():
            files = sorted(path.glob("*.npy"))
        elif path.is_file():
            files = [path]
        else:
            raise FileNotFoundError(path)

        if not files:
            raise FileNotFoundError(
                f"No NumPy shards found at {path}"
            )

        return files

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, index):

        if index < 0:
            index += self.total_sequences

        if index < 0 or index >= self.total_sequences:
            raise IndexError(index)

        shard_index = bisect.bisect_right(
            self.cumulative_ends, index
        )

        previous_end = (
            self.cumulative_ends[shard_index - 1]
            if shard_index > 0 else 0
        )

        local_index = index - previous_end

        x = self.inputs[shard_index][local_index]
        y = self.targets[shard_index][local_index]

        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))

        return x, y


# ------------------------------------------------------------
# Checkpoint utilities
# ------------------------------------------------------------

CHECKPOINT_FORMAT = "custom_llm_v3"


def save_v3_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    step,
    config,
):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "format": CHECKPOINT_FORMAT,
        "step": step,
        "config": config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None
        ),
    }

    temporary_path = path.with_name(path.name + ".tmp")

    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def load_v3_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    model_config,
):

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            "This is not a V3 checkpoint. "
            "V1/V2 checkpoints cannot be used for V3."
        )

    saved_model_config = state["config"]["model"]

    if saved_model_config != model_config:
        raise ValueError(
            "Checkpoint architecture does not match current V3 config."
        )

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])

    if "scaler" in state:
        scaler.load_state_dict(state["scaler"])

    torch.set_rng_state(state["torch_rng_state"])

    np.random.set_state(state["numpy_rng_state"])
    random.setstate(state["python_rng_state"])

    if (
        torch.cuda.is_available()
        and state.get("cuda_rng_state") is not None
    ):
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])

    return int(state["step"])


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate(model, loader, device, vocab_size, use_amp):

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.inference_mode():

        for inputs, targets in loader:

            inputs = inputs.to(
                device, non_blocking=True
            )

            targets = targets.to(
                device, non_blocking=True
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):

                logits = model(inputs)

                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    targets.reshape(-1),
                    reduction="sum",
                )

            total_loss += loss.item()
            total_tokens += targets.numel()

    if was_training:
        model.train()

    if total_tokens == 0:
        raise ValueError("Evaluation loader is empty")

    return total_loss / total_tokens


def build_eval_loader(
    dataset,
    batch_size,
    eval_samples,
    seed,
    device,
):

    count = min(eval_samples, len(dataset))

    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        len(dataset),
        generator=generator
    )[:count].tolist()

    subset = Subset(dataset, indices)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )


# ------------------------------------------------------------
# Main training function
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Custom-LLM V3 training"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/100m_v3.yaml",
    )

    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(
            "/content/drive/MyDrive/Custom-LLM-V2/"
            "checkpoints_v3/100m_v3_latest.pt"
        ),
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path(
            "/content/drive/MyDrive/Custom-LLM-V2/"
            "logs_v3/100m_v3_training.log"
        ),
    )

    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--no-amp", action="store_true")

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]

    total_steps = (
        args.steps
        if args.steps is not None
        else training_config["steps"]
    )

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else training_config["batch_size"]
    )

    log_every = (
        args.log_every
        if args.log_every is not None
        else training_config["log_every"]
    )

    checkpoint_every = (
        args.checkpoint_every
        if args.checkpoint_every is not None
        else training_config.get("checkpoint_every", 500)
    )

    eval_every = (
        args.eval_every
        if args.eval_every is not None
        else training_config.get("eval_every", 500)
    )

    eval_samples = (
        args.eval_samples
        if args.eval_samples is not None
        else training_config.get("eval_samples", 512)
    )

    full_eval_every = training_config.get("full_eval_every", 0)

    if min(
        total_steps,
        batch_size,
        log_every,
        checkpoint_every,
        eval_every,
        eval_samples,
    ) <= 0:
        raise ValueError("Training and evaluation settings must be positive")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = (
        device.type == "cuda"
        and not args.no_amp
    )

    print("=" * 70)
    print("CUSTOM-LLM V3")
    print("=" * 70)

    print("Device:", device)
    print("Mixed precision:", use_amp)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Batch size:", batch_size)
    print("Total steps:", total_steps)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TransformerLM(
        vocab_size=model_config["vocab_size"],
        d_model=model_config["d_model"],
        n_layers=model_config["n_layers"],
        n_heads=model_config["n_heads"],
        ffn_hidden_size=model_config["hidden_dim"],
        max_seq_len=model_config["context_length"],
        dropout=model_config["dropout"],
    ).to(device)

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Model parameters: {total_params:,}")

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    train_dataset = V3TokenDataset(
        data_config["train_inputs"],
        data_config["train_targets"],
        model_config["context_length"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    print(f"Training sequences: {len(train_dataset):,}")

    # --------------------------------------------------------
    # Separate validation datasets
    # --------------------------------------------------------

    validation_root = (
        Path(data_config["val_inputs"]).parent
    )

    fineweb_val = V3TokenDataset(
        validation_root / "inputs/fineweb_validation_00000.npy",
        validation_root / "targets/fineweb_validation_00000.npy",
        model_config["context_length"],
    )

    wikipedia_val = V3TokenDataset(
        validation_root / "inputs/wikipedia_validation_00000.npy",
        validation_root / "targets/wikipedia_validation_00000.npy",
        model_config["context_length"],
    )

    print(f"FineWeb validation: {len(fineweb_val):,}")
    print(f"Wikipedia validation: {len(wikipedia_val):,}")

    fineweb_loader = build_eval_loader(
        fineweb_val,
        batch_size,
        eval_samples,
        1234,
        device,
    )

    wikipedia_loader = build_eval_loader(
        wikipedia_val,
        batch_size,
        eval_samples,
        5678,
        device,
    )

    full_fineweb_loader = DataLoader(
        fineweb_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    full_wikipedia_loader = DataLoader(
        wikipedia_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"FineWeb evaluation subset: "
        f"{len(fineweb_loader.dataset):,}"
    )

    print(
        f"Wikipedia evaluation subset: "
        f"{len(wikipedia_loader.dataset):,}"
    )

    # --------------------------------------------------------
    # Optimizer and scheduler
    # --------------------------------------------------------

    optimizer = build_optimizer(
        model,
        training_config["learning_rate"],
        training_config["weight_decay"],
    )

    scheduler = build_scheduler(
        optimizer,
        training_config["warmup_steps"],
        total_steps,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    start_step = 0

    if args.resume_from is not None:

        start_step = load_v3_checkpoint(
            args.resume_from,
            model,
            optimizer,
            scheduler,
            scaler,
            model_config,
        )

        if start_step >= total_steps:
            raise ValueError(
                f"Checkpoint step {start_step} "
                f"has already reached target {total_steps}"
            )

        print(f"Resuming from V3 step {start_step}")

    else:
        print("Starting from fresh model initialization.")

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    args.checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Evaluation helper
    # --------------------------------------------------------

    def run_validation(step, log_file):

        fineweb_loss = evaluate(
            model,
            fineweb_loader,
            device,
            model_config["vocab_size"],
            use_amp,
        )

        wikipedia_loss = evaluate(
            model,
            wikipedia_loader,
            device,
            model_config["vocab_size"],
            use_amp,
        )

        # Weight by actual validation dataset sizes.
        combined_loss = (
            fineweb_loss * len(fineweb_val)
            + wikipedia_loss * len(wikipedia_val)
        ) / (
            len(fineweb_val) + len(wikipedia_val)
        )

        messages = [
            (
                f"VALIDATION step={step} "
                f"fineweb_loss={fineweb_loss:.5f} "
                f"fineweb_ppl={np.exp(min(fineweb_loss, 20)):.4f}"
            ),
            (
                f"VALIDATION step={step} "
                f"wikipedia_loss={wikipedia_loss:.5f} "
                f"wikipedia_ppl={np.exp(min(wikipedia_loss, 20)):.4f}"
            ),
            (
                f"VALIDATION step={step} "
                f"combined_weighted_loss={combined_loss:.5f} "
                f"combined_ppl={np.exp(min(combined_loss, 20)):.4f}"
            ),
        ]

        for message in messages:
            print(message)
            log_file.write(message + "\n")

        log_file.flush()

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    model.train()

    data_iterator = iter(train_loader)

    running_loss = 0.0
    running_steps = 0

    print("\nStarting training...\n")

    with args.log_file.open(
        "a",
        encoding="utf-8",
        buffering=1,
    ) as log_file:

        for step in range(start_step + 1, total_steps + 1):

            try:
                inputs, targets = next(data_iterator)

            except StopIteration:
                data_iterator = iter(train_loader)
                inputs, targets = next(data_iterator)

            inputs = inputs.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):

                logits = model(inputs)

                loss = F.cross_entropy(
                    logits.reshape(
                        -1,
                        model_config["vocab_size"],
                    ),
                    targets.reshape(-1),
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at step {step}: "
                    f"{loss.item()}"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            previous_scale = scaler.get_scale()

            scaler.step(optimizer)
            scaler.update()

            current_scale = scaler.get_scale()

            # Do not advance LR schedule if GradScaler skipped
            # an optimizer update due to non-finite gradients.
            optimizer_step_skipped = (
                use_amp and current_scale < previous_scale
            )

            if not optimizer_step_skipped:
                scheduler.step()

            running_loss += loss.item()
            running_steps += 1

            # ------------------------------------------------
            # Training logs
            # ------------------------------------------------

            if (
                step == start_step + 1
                or step % log_every == 0
                or step == total_steps
            ):

                average_loss = (
                    running_loss / max(running_steps, 1)
                )

                learning_rate = optimizer.param_groups[0]["lr"]

                message = (
                    f"step={step}/{total_steps} "
                    f"train_loss={average_loss:.5f} "
                    f"lr={learning_rate:.8f} "
                    f"grad_norm={float(gradient_norm):.5f} "
                    f"scale={scaler.get_scale():.1f} "
                    f"optimizer_skipped={optimizer_step_skipped}"
                )

                print(message)
                log_file.write(message + "\n")
                log_file.flush()

                running_loss = 0.0
                running_steps = 0

            # ------------------------------------------------
            # Checkpoint
            # ------------------------------------------------

            if (
                step % checkpoint_every == 0
                or step == total_steps
            ):

                save_v3_checkpoint(
                    args.checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                    config,
                )

                message = (
                    f"CHECKPOINT saved step={step} "
                    f"path={args.checkpoint_path}"
                )

                print(message)
                log_file.write(message + "\n")
                log_file.flush()

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            if (
                step % eval_every == 0
                or step == total_steps
            ):

                run_validation(step, log_file)
                model.train()

                # Run complete validation at the final step,
                # or at the configured full evaluation interval.
                if (
                    step == total_steps
                    or (
                        full_eval_every > 0
                        and step % full_eval_every == 0
                    )
                ):

                    print("Running FULL validation...")

                    full_fw_loss = evaluate(
                        model,
                        full_fineweb_loader,
                        device,
                        model_config["vocab_size"],
                        use_amp,
                    )

                    full_wiki_loss = evaluate(
                        model,
                        full_wikipedia_loader,
                        device,
                        model_config["vocab_size"],
                        use_amp,
                    )

                    full_combined_loss = (
                        full_fw_loss * len(fineweb_val)
                        + full_wiki_loss * len(wikipedia_val)
                    ) / (
                        len(fineweb_val) + len(wikipedia_val)
                    )

                    full_messages = [
                        f"FULL_VALIDATION step={step} "
                        f"fineweb_loss={full_fw_loss:.5f}",

                        f"FULL_VALIDATION step={step} "
                        f"wikipedia_loss={full_wiki_loss:.5f}",

                        f"FULL_VALIDATION step={step} "
                        f"combined_weighted_loss={full_combined_loss:.5f} "
                        f"combined_ppl={np.exp(min(full_combined_loss, 20)):.4f}",
                    ]

                    for message in full_messages:
                        print(message)
                        log_file.write(message + "\n")

                    log_file.flush()
                    model.train()

    print("\nTraining completed.")
    print("Final checkpoint:", args.checkpoint_path)
    print("Training log:", args.log_file)


if __name__ == "__main__":
    main()