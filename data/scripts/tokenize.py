"""Tokenize the deduplicated corpus into fixed-length causal-LM examples."""
from pathlib import Path
import argparse
import json

import numpy as np
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_CORPUS = PROJECT_ROOT / "data/cleaned/fineweb_deduplicated.jsonl"
DEFAULT_TOKENIZER = PROJECT_ROOT / "tokenizer/tokenizer.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/tokenized"
DEFAULT_BLOCK_SIZE = 512


def read_token_ids(corpus_file: Path, tokenizer: Tokenizer, eos_id: int) -> list[int]:
    token_ids: list[int] = []
    with corpus_file.open(encoding="utf-8") as corpus:
        for line in corpus:
            record = json.loads(line)
            text = record.get("text", "")
            if text:
                token_ids.extend(tokenizer.encode(text).ids)
                token_ids.append(eos_id)
    return token_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    args = parser.parse_args()

    if args.block_size < 1:
        raise ValueError("--block-size must be positive")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eos_id = tokenizer.token_to_id("<EOS>")
    if eos_id is None:
        raise ValueError("Tokenizer must contain an <EOS> token")

    print(f"Tokenizing {args.input}...")
    token_ids = read_token_ids(args.input, tokenizer, eos_id)
    sequence_count = (len(token_ids) - 1) // args.block_size
    usable_tokens = sequence_count * args.block_size
    if sequence_count == 0:
        raise ValueError("Corpus does not contain enough tokens for one sequence")

    input_ids = np.asarray(token_ids[:usable_tokens], dtype=np.uint16).reshape(sequence_count, args.block_size)
    target_ids = np.asarray(token_ids[1 : usable_tokens + 1], dtype=np.uint16).reshape(sequence_count, args.block_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_file = args.output_dir / "fineweb_train_inputs.npy"
    target_file = args.output_dir / "fineweb_train_targets.npy"
    metadata_file = args.output_dir / "fineweb_train_metadata.json"
    np.save(input_file, input_ids)
    np.save(target_file, target_ids)
    metadata_file.write_text(
        json.dumps(
            {
                "source": str(args.input),
                "tokenizer": str(args.tokenizer),
                "vocab_size": tokenizer.get_vocab_size(),
                "eos_id": eos_id,
                "block_size": args.block_size,
                "sequence_count": sequence_count,
                "discarded_tokens": len(token_ids) - usable_tokens - 1,
                "dtype": "uint16",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Total tokens: {len(token_ids):,}")
    print(f"Training sequences: {sequence_count:,} x {args.block_size} tokens")
    print(f"Saved inputs to {input_file}")
    print(f"Saved targets to {target_file}")
    print(f"Saved metadata to {metadata_file}")


if __name__ == "__main__":
    main()