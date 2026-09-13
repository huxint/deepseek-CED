"""Generate text from a tiny CED checkpoint using explicit incremental KV caches."""

import argparse
import logging

import torch

from .config import ModelConfig
from .model import CEDLanguageModel, PrefillMode
from .runtime import autocast_context, load_checkpoint, resolve_device, resolve_precision
from .tokenizer import ByteTokenizer

logger = logging.getLogger(__name__)


def sample_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    scores = logits.float() / temperature
    if top_k > 0:
        cutoff = scores.topk(min(top_k, scores.size(-1))).values[:, -1:]
        scores = scores.masked_fill(scores < cutoff, -torch.inf)
    return torch.multinomial(scores.softmax(dim=-1), num_samples=1)


@torch.inference_mode()
def generate(
    model: CEDLanguageModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    prefill: PrefillMode = "exact",
) -> torch.Tensor:
    """Generate for one prompt; model must be in eval mode, with a nonempty [1, T] input."""
    if input_ids.size(1) + max_new_tokens > model.config.max_seq_len:
        raise ValueError(
            f"prompt ({input_ids.size(1)} tokens) + generation ({max_new_tokens}) exceeds "
            f"max_seq_len={model.config.max_seq_len}; shorten the prompt or --max-new-tokens"
        )
    if max_new_tokens == 0:
        return input_ids.clone()
    result = model.prefill(input_ids, mode=prefill)
    pieces = [input_ids]
    for index in range(max_new_tokens):
        scores = result.logits[:, -1].clone()
        scores[:, ByteTokenizer.bos_id] = -torch.inf
        token = sample_token(scores, temperature, top_k)
        pieces.append(token)
        if token.item() == ByteTokenizer.eos_id or index == max_new_tokens - 1:
            break
        result = model.decode(token, result.cache)
    return torch.cat(pieces, dim=1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Question: What is 2 plus 3?\nAnswer:")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 for greedy decoding")
    parser.add_argument("--top-k", type=int, default=40, help="0 disables top-k filtering")
    parser.add_argument("--prefill", choices=("exact", "bounded"), default="exact")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    checkpoint = load_checkpoint(args.checkpoint)
    config = ModelConfig(**checkpoint["model_config"])
    if config.vocab_size != ByteTokenizer.vocab_size:
        raise ValueError("checkpoint is incompatible with the byte tokenizer")
    model = CEDLanguageModel(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    del checkpoint
    tokenizer = ByteTokenizer()
    tokens = torch.tensor(
        [tokenizer.encode(args.prompt, bos=True)], dtype=torch.long, device=device
    )
    if args.prefill == "bounded" and tokens.size(1) > config.window_size:
        logger.info("Using approximate decoder SWA bounded replay.")
    with autocast_context(device, precision):
        output = generate(
            model,
            tokens,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.prefill,
        )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
