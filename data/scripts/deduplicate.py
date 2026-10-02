"""Remove duplicate lines while preserving their original order."""
from pathlib import Path
import argparse


def deduplicate(text: str) -> str:
    seen: set[str] = set()
    unique = []
    for line in text.splitlines():
        normalized = line.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return "\n".join(unique) + ("\n" if unique else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, default=Path(__file__).parents[1] / "cleaned/filtered.txt", nargs="?")
    parser.add_argument("output", type=Path, default=Path(__file__).parents[1] / "cleaned/deduplicated.txt", nargs="?")
    args = parser.parse_args()
    args.output.write_text(deduplicate(args.input.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()
