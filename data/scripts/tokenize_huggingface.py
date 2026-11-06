"""Stream Hugging Face text datasets into fixed-size NumPy training shards."""
from __future__ import annotations

import argparse
from itertools import cycle
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer


def parse_source(value: str) -> tuple[str, str | None]:
    dataset_id, separator, config = value.partition(":")
    if not dataset_id or (separator and not config):
        raise ValueError(f"Invalid source {value!r}; use DATASET_ID[:CONFIG]")
    return dataset_id, config or None


def documents(dataset_id: str, config: str | None, split: str):
    kwargs = {"path": dataset_id, "split": split, "streaming": True}
    if config is not None:
        kwargs["name"] = config
    return load_dataset(**kwargs)


def write_shard(tokens: list[int], output_dir: Path, shard_index: int, block_size: int) -> int:
    sequence_count = (len(tokens) - 1) // block_size
    if sequence_count == 0:
        return 0
    usable_tokens = sequence_count * block_size
    inputs = np.asarray(tokens[:usable_tokens], dtype=np.uint16).reshape(-1, block_size)
    targets = np.asarray(tokens[1 : usable_tokens + 1], dtype=np.uint16).reshape(-1, block_size)
    np.save(output_dir / "inputs" / f"shard_{shard_index:05d}.npy", inputs)
    np.save(output_dir / "targets" / f"shard_{shard_index:05d}.npy", targets)
    return sequence_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="DATASET_ID[:CONFIG], repeat for an equal document mix")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tokenizer.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/tokenized/100m"))
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--shard-sequences", type=int, default=10_000)
    parser.add_argument("--max-tokens", type=int, default=0, help="Stop after this many tokens; 0 means no limit")
    args = parser.parse_args()
    if args.block_size < 1 or args.shard_sequences < 1:
        raise ValueError("--block-size and --shard-sequences must be positive")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eos_id = tokenizer.token_to_id("<EOS>")
    if eos_id is None:
        raise ValueError("Tokenizer must contain an <EOS> token")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "inputs").mkdir(exist_ok=True)
    (args.output_dir / "targets").mkdir(exist_ok=True)

    streams = [
        iter(documents(*parse_source(source), args.split))
        for source in args.source
    ]
    token_buffer: list[int] = []
    total_tokens = 0
    total_sequences = 0
    shard_index = 0
    exhausted = [False] * len(streams)

    for stream_index in cycle(range(len(streams))):
        if all(exhausted) or (args.max_tokens and total_tokens >= args.max_tokens):
            break
        if exhausted[stream_index]:
            continue
        try:
            record = next(streams[stream_index])
        except StopIteration:
            exhausted[stream_index] = True
            continue
        text = record.get("text", "")
        if not isinstance(text, str) or not text:
            continue
        token_buffer.extend(tokenizer.encode(text).ids)
        token_buffer.append(eos_id)
        if args.max_tokens:
            token_buffer = token_buffer[: max(0, args.max_tokens - total_tokens)]
        while (len(token_buffer) - 1) // args.block_size >= args.shard_sequences:
            shard_tokens = token_buffer[: args.shard_sequences * args.block_size + 1]
            count = write_shard(shard_tokens, args.output_dir, shard_index, args.block_size)
            total_sequences += count
            total_tokens += count * args.block_size
            shard_index += 1
            del token_buffer[: args.shard_sequences * args.block_size]

    if (len(token_buffer) - 1) // args.block_size:
        count = write_shard(token_buffer, args.output_dir, shard_index, args.block_size)
        total_sequences += count
        total_tokens += count * args.block_size
    print(f"Saved {total_sequences:,} sequences ({total_tokens:,} tokens) to {args.output_dir}")


if __name__ == "__main__":
    main()
