"""Clean the sampled FineWeb JSONL corpus."""
from pathlib import Path
import json
import re
import unicodedata


PROJECT_ROOT = Path(__file__).parents[2]
INPUT_FILE = PROJECT_ROOT / "data/raw/fineweb/fineweb_sample.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data/cleaned/fineweb_cleaned.jsonl"
MIN_TEXT_LENGTH = 20


def is_obviously_broken(text: str) -> bool:
    """Reject common extraction failures without filtering normal punctuation."""
    if len(text) < MIN_TEXT_LENGTH or "\ufffd" in text:
        return True

    control_characters = sum(
        unicodedata.category(character) == "Cc" and character not in "\n\r\t"
        for character in text
    )
    return control_characters / len(text) > 0.01


def clean_text(text: str) -> str | None:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or is_obviously_broken(text):
        return None
    return text


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    written = 0

    with INPUT_FILE.open(encoding="utf-8") as source, OUTPUT_FILE.open("w", encoding="utf-8") as destination:
        for line in source:
            total += 1
            record = json.loads(line)
            text = clean_text(record.get("text", ""))
            if text is None:
                continue
            destination.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1

    print(f"Cleaned {written:,} of {total:,} documents")
    print(f"Saved cleaned corpus to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
