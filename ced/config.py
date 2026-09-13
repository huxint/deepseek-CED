import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int = 258
    d_model: int = 64
    n_heads: int = 4
    n_kv_heads: int = 2
    ffn_dim: int = 128
    encoder_layers: int = 2
    decoder_layers: int = 2
    encoder_swa_only_layers: int = 1
    decoder_kv_groups: int = 1
    window_size: int = 32
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
