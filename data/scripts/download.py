from pathlib import Path
import json

from datasets import load_dataset


# Configuration
NUM_DOCUMENTS = 10_000
OUTPUT_DIR = Path("data/raw/fineweb")
OUTPUT_FILE = OUTPUT_DIR / "fineweb_sample.jsonl"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FineWeb-Edu...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="CC-MAIN-2024-10",
        split="train",
        streaming=True,
    )

    print(f"Collecting {NUM_DOCUMENTS:,} documents...")
    count = 0
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for example in dataset:
            text = example.get("text", "").strip()
            if not text:
                continue

            file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1

            if count % 1_000 == 0:
                print(f"Collected {count:,} documents...")
            if count >= NUM_DOCUMENTS:
                break

    print(f"Saved {count:,} documents to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()