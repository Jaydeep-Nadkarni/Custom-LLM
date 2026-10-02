"""Normalize raw text files into one UTF-8 document."""
from pathlib import Path
import re

RAW_DIR = Path(__file__).parents[1] / "raw"
CLEANED_DIR = Path(__file__).parents[1] / "cleaned"


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted(RAW_DIR.glob("*.txt"))
    combined = "\n\n".join(clean_text(path.read_text(encoding="utf-8")) for path in sources)
    (CLEANED_DIR / "corpus.txt").write_text(combined + "\n", encoding="utf-8")
    print(f"Wrote {len(combined):,} characters from {len(sources)} files")


if __name__ == "__main__":
    main()
