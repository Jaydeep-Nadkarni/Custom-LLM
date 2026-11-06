# my-llm

A small, educational decoder-only Transformer project. The repository is organized so data preparation, tokenization, modeling, training, evaluation, and inference can evolve independently.

## Setup

```powershell
cd my-llm
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Project flow

1. Put source text files in `data/raw/`.
2. Run the scripts in `data/scripts/` to clean, filter, and deduplicate data.
3. Train a tokenizer with `python tokenizer/train_tokenizer.py`.
4. For large Hugging Face datasets, stream and shard them without downloading the corpus:
   `python data/scripts/tokenize_huggingface.py --source HuggingFaceFW/fineweb-edu:default --source wikimedia/wikipedia:20231101.en --max-tokens 2000000000`.
5. Adjust a config in `configs/` and train with `python -m training.train --config configs/100m.yaml`.
6. Generate text with `python inference/generate.py --checkpoint checkpoints/latest.pt --prompt "Hello"`.

The model is a learning scaffold, not a production LLM. The 100M configuration expects
the sharded files in `data/tokenized/100m/`. The Hugging Face script uses
streaming and writes only tokenized NumPy shards locally.
