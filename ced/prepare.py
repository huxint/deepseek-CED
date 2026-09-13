"""Prepare little-endian uint16 token streams using only the Python standard library."""

import argparse
import hashlib
import json
import logging
import random
import sys
from array import array
from pathlib import Path

from .metadata import CorpusMetadata, TokenMetadata
from .tokenizer import ByteTokenizer

logger = logging.getLogger(__name__)


def demo_documents() -> list[str]:
    """Original synthetic text for checking the training pipeline, not a real pretraining corpus."""
    rng = random.Random(2026)
    colors = ["red", "blue", "green", "yellow"]
    animals = ["cat", "dog", "bird", "fish"]
    documents = []
    for _ in range(1200):
        a, b = rng.randrange(20), rng.randrange(20)
        color, animal = rng.choice(colors), rng.choice(animals)
        documents.append(
            f"Question: What is {a} plus {b}?\nAnswer: {a} plus {b} is {a + b}.\n"
            f"The {animal} is {color}. What color is the {animal}? It is {color}.\n"
            "小猫坐在窗边，看着外面的树。太阳升起，新的一天开始了。\n"
            "问：模型怎样学习？\n答：模型读取前面的文字，预测下一个符号。\n"
        )
    return documents


def read_documents(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        documents = []
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["text"].strip():
                    documents.append(record["text"])
        return documents
    return [part.strip() for part in path.read_text(encoding="utf-8").split("\n\n") if part.strip()]


def write_tokens(path: Path, documents: list[str]) -> TokenMetadata:
    tokenizer = ByteTokenizer()
    digest = hashlib.sha256()
    count = 0
    with path.open("wb") as output:
        for document in documents:
            tokens = array("H", tokenizer.encode(document, bos=True, eos=True))
            if sys.byteorder != "little":
                tokens.byteswap()
            chunk = tokens.tobytes()
            output.write(chunk)
            digest.update(chunk)
            count += len(tokens)
    return {"tokens": count, "sha256": digest.hexdigest()}


def prepare(
    documents: list[str],
    out_dir: Path,
    val_fraction: float = 0.1,
    seed: int = 1337,
) -> CorpusMetadata:
    documents = [doc for doc in documents if doc.strip()]
    if not documents:
        raise ValueError("the corpus contains no text")
    if len(documents) == 1:
        text = documents[0]
        boundary = int(len(text) * (1 - val_fraction))
        train_docs, val_docs = [text[:boundary]], [text[boundary:]]
        split = "contiguous text split before tokenization"
    else:
        documents = documents.copy()
        random.Random(seed).shuffle(documents)
        val_count = min(len(documents) - 1, max(1, int(len(documents) * val_fraction)))
        train_docs, val_docs = documents[val_count:], documents[:val_count]
        split = "shuffled document split before tokenization"
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata: CorpusMetadata = {
        "format_version": 1,
        "tokenizer": ByteTokenizer.name,
        "vocab_size": ByteTokenizer.vocab_size,
        "dtype": "uint16-le",
        "seed": seed,
        "split": split,
        "train": write_tokens(out_dir / "train.bin", train_docs),
        "val": write_tokens(out_dir / "val.bin", val_docs),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input", type=Path, nargs="+", help="UTF-8 .txt or .jsonl files (text field)"
    )
    source.add_argument(
        "--demo", action="store_true", help="create a small synthetic smoke-test corpus"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    documents = (
        demo_documents()
        if args.demo
        else [doc for path in args.input for doc in read_documents(path)]
    )
    metadata = prepare(documents, args.out_dir, args.val_fraction, args.seed)
    logger.info(
        "Prepared %s: train=%s tokens, val=%s tokens; vocabulary=%d",
        args.out_dir,
        f"{metadata['train']['tokens']:,}",
        f"{metadata['val']['tokens']:,}",
        metadata["vocab_size"],
    )


if __name__ == "__main__":
    main()
