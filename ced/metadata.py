"""The small, versioned on-disk corpus format."""

from typing import Literal, TypedDict

type Split = Literal["train", "val"]


class TokenMetadata(TypedDict):
    tokens: int
    sha256: str


class CorpusMetadata(TypedDict):
    format_version: int
    tokenizer: str
    vocab_size: int
    dtype: str
    seed: int
    split: str
    train: TokenMetadata
    val: TokenMetadata
