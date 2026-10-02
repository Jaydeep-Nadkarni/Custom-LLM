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
4. Adjust a config in `configs/` and train with `python -m training.train --config configs/10m.yaml`.
5. Generate text with `python inference/generate.py --checkpoint checkpoints/latest.pt --prompt "Hello"`.

The model is a learning scaffold, not a production LLM. Start with the 10m configuration and a small corpus before scaling up.
