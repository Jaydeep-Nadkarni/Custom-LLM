import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import DecoderLM
from inference.sampling import sample_next_token


def generate(model, prompt_ids: list[int], max_new_tokens: int, temperature: float, top_k: int | None) -> list[int]:
    tokens = torch.tensor([prompt_ids], dtype=torch.long)
    for _ in range(max_new_tokens):
        logits, _ = model(tokens)
        next_token = sample_next_token(logits[:, -1, :], temperature, top_k)
        tokens = torch.cat((tokens, next_token), dim=1)
    return tokens[0].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pt"))
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DecoderLM(**state["config"])
    model.load_state_dict(state["model"])
    model.eval()
    prompt_ids = tokenizer.encode(args.prompt).ids
    output_ids = generate(model, prompt_ids, args.max_new_tokens, args.temperature, args.top_k)
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
