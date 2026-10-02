"""Train a byte-level BPE tokenizer from the deduplicated JSONL corpus."""
from pathlib import Path
import json

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


PROJECT_ROOT = Path(__file__).parents[1]
CORPUS_FILE = PROJECT_ROOT / "data/cleaned/fineweb_deduplicated.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "tokenizer"
VOCAB_SIZE = 32_000
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


def read_documents():
    with CORPUS_FILE.open(encoding="utf-8") as corpus:
        for line in corpus:
            record = json.loads(line)
            text = record.get("text", "")
            if text:
                yield text


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(read_documents(), trainer=trainer)
    tokenizer.save(str(OUTPUT_DIR / "tokenizer.json"))

    vocabulary = tokenizer.get_vocab()
    vocabulary = dict(sorted(vocabulary.items(), key=lambda item: item[1]))
    (OUTPUT_DIR / "vocab.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Trained tokenizer with {tokenizer.get_vocab_size():,} tokens")
    print(f"Saved tokenizer to {OUTPUT_DIR / 'tokenizer.json'}")
    print(f"Saved vocabulary to {OUTPUT_DIR / 'vocab.json'}")


if __name__ == "__main__":
    main()
