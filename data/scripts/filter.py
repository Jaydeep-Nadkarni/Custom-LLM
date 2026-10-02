"""Keep non-empty lines above a minimum length."""
from pathlib import Path
import argparse


def filter_text(text: str, min_chars: int) -> str:
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= min_chars]
    return "\n".join(lines) + ("\n" if lines else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, default=Path(__file__).parents[1] / "cleaned/corpus.txt", nargs="?")
    parser.add_argument("output", type=Path, default=Path(__file__).parents[1] / "cleaned/filtered.txt", nargs="?")
    parser.add_argument("--min-chars", type=int, default=20)
    args = parser.parse_args()
    args.output.write_text(filter_text(args.input.read_text(encoding="utf-8"), args.min_chars), encoding="utf-8")


if __name__ == "__main__":
    main()
