"""Remove exact duplicate documents from the cleaned JSONL corpus."""
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).parents[2]
INPUT_FILE = PROJECT_ROOT / "data/cleaned/fineweb_cleaned.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data/cleaned/fineweb_deduplicated.jsonl"


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    documents_before = 0
    documents_after = 0
    seen: set[str] = set()

    with INPUT_FILE.open(encoding="utf-8") as source, OUTPUT_FILE.open("w", encoding="utf-8") as destination:
        for line in source:
            documents_before += 1
            record = json.loads(line)
            text = record.get("text", "")
            if text in seen:
                continue

            seen.add(text)
            destination.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            documents_after += 1

    duplicates_removed = documents_before - documents_after
    print(f"Documents before: {documents_before:,}")
    print(f"Documents after: {documents_after:,}")
    print(f"Duplicates removed: {duplicates_removed:,}")
    print(f"Saved deduplicated corpus to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
