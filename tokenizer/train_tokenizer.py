"""Train a byte-level BPE tokenizer from the cleaned corpus."""
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).parents[1]


def main() -> None:
    corpus = ROOT / "data/cleaned/deduplicated.txt"
    output = ROOT / "tokenizer"
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(vocab_size=16_000, min_frequency=2, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])
    tokenizer.train([str(corpus)], trainer)
    tokenizer.save(str(output / "tokenizer.json"))
    tokenizer.model.save(str(output))


if __name__ == "__main__":
    main()
