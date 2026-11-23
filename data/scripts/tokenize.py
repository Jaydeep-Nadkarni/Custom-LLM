from pathlib import Path
import argparse
import json

import numpy as np
from tokenizers import Tokenizer


def tokenize_file(input_file, tokenizer, output_dir,
                  block_size=512, shard_sequences=10000):

    input_file = Path(input_file)
    output_dir = Path(output_dir)

    input_dir = output_dir / "inputs"
    target_dir = output_dir / "targets"
    metadata_dir = output_dir / "metadata"

    for folder in (input_dir, target_dir, metadata_dir):
        folder.mkdir(parents=True, exist_ok=True)

    eos_id = tokenizer.token_to_id("<EOS>")
    if eos_id is None:
        raise ValueError("Tokenizer must contain <EOS>")

    if tokenizer.get_vocab_size() > 65536:
        raise ValueError("Vocabulary exceeds uint16 capacity")

    prefix = input_file.stem

    x_batch = np.empty((shard_sequences, block_size), dtype=np.uint16)
    y_batch = np.empty((shard_sequences, block_size), dtype=np.uint16)

    pending = []
    position = 0
    batch_count = 0
    shard_count = 0
    total_tokens = 0
    total_sequences = 0
    documents = 0

    def save_shard(count):
        nonlocal shard_count

        if count == 0:
            return

        np.save(
            input_dir / f"{prefix}_{shard_count:05d}.npy",
            x_batch[:count]
        )
        np.save(
            target_dir / f"{prefix}_{shard_count:05d}.npy",
            y_batch[:count]
        )

        shard_count += 1

    def consume():
        nonlocal position, batch_count, total_sequences

        while len(pending) - position >= block_size + 1:
            x_batch[batch_count] = pending[position:position + block_size]
            y_batch[batch_count] = pending[position + 1:position + block_size + 1]

            position += block_size
            batch_count += 1
            total_sequences += 1

            if batch_count == shard_sequences:
                save_shard(batch_count)
                batch_count = 0

        # Compact occasionally, not after every sequence.
        if position >= 100000:
            del pending[:position]
            position = 0

    print(f"Processing: {input_file}")

    with input_file.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            text = record.get("text", "")

            if not isinstance(text, str) or not text.strip():
                continue

            ids = tokenizer.encode(text).ids
            pending.extend(ids)
            pending.append(eos_id)

            total_tokens += len(ids) + 1
            documents += 1

            consume()

            if documents % 10000 == 0:
                print(
                    f"Documents: {documents:,} | "
                    f"Sequences: {total_sequences:,}"
                )

    # Remaining complete sequences
    consume()
    save_shard(batch_count)

    metadata = {
        "source": str(input_file),
        "vocab_size": tokenizer.get_vocab_size(),
        "eos_id": eos_id,
        "block_size": block_size,
        "sequence_count": total_sequences,
        "total_tokens": total_tokens,
        "shard_count": shard_count,
        "documents": documents,
        "dtype": "uint16"
    }

    metadata_path = metadata_dir / f"{prefix}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nCompleted: {prefix}")
    print(f"Documents: {documents:,}")
    print(f"Tokens: {total_tokens:,}")
    print(f"Sequences: {total_sequences:,}")
    print(f"Shards: {shard_count}")

    return metadata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--shard-sequences", type=int, default=10000)

    args = parser.parse_args()

    if args.block_size < 1 or args.shard_sequences < 1:
        raise ValueError("Invalid block or shard size")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    tokenize_file(
        args.input,
        tokenizer,
        args.output_dir,
        args.block_size,
        args.shard_sequences
    )


if __name__ == "__main__":
    main()
