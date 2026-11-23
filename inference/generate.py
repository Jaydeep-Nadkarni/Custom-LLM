import argparse
from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.model import TransformerLM
from inference.sampling import sample_next_token


def build_model_from_config(model_config: dict) -> TransformerLM:
    return TransformerLM(
        vocab_size=model_config["vocab_size"],
        d_model=model_config["d_model"],
        n_layers=model_config["n_layers"],
        n_heads=model_config["n_heads"],
        ffn_hidden_size=model_config["hidden_dim"],
        max_seq_len=model_config["context_length"],
        dropout=model_config["dropout"],
    )


def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    eos_id: int | None = None,
) -> list[int]:
    """Autoregressive sampling. Returns prompt + generated token IDs.

    The context is cropped to the model's maximum sequence length, and
    generation stops early when ``eos_id`` is sampled (EOS is not appended).
    """
    device = next(model.parameters()).device
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model(tokens[:, -model.max_seq_len:])
            next_token = sample_next_token(logits[:, -1, :], temperature, top_k)
            if eos_id is not None and next_token.item() == eos_id:
                break
            tokens = torch.cat((tokens, next_token), dim=1)
    return tokens[0].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tokenizer_v3.json"))
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_config(state["config"]["model"])
    model.load_state_dict(state["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    prompt_ids = tokenizer.encode(args.prompt).ids
    output_ids = generate(
        model,
        prompt_ids,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        eos_id=tokenizer.token_to_id("<EOS>"),
    )
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
