import hashlib
import json
from pathlib import Path

import torch

from .metadata import CorpusMetadata, Split
from .tokenizer import ByteTokenizer


class TokenCorpus:
    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        self.metadata: CorpusMetadata = json.loads(
            (directory / "meta.json").read_text(encoding="utf-8")
        )
        if (
            self.metadata.get("format_version") != 1
            or self.metadata.get("tokenizer") != ByteTokenizer.name
            or self.metadata.get("vocab_size") != ByteTokenizer.vocab_size
            or self.metadata.get("dtype") != "uint16-le"
        ):
            raise ValueError("unsupported data format or tokenizer; run python -m ced.prepare")
        self.streams: dict[Split, torch.Tensor] = {}
        splits: tuple[Split, ...] = ("train", "val")
        for split in splits:
            path = directory / f"{split}.bin"
            expected = self.metadata[split]
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if path.stat().st_size != expected["tokens"] * 2 or digest != expected["sha256"]:
                raise ValueError(f"{path} does not match meta.json; prepare the data again")
            # IDs are 0..257, so signed int16 and uint16 have identical representations.
            self.streams[split] = torch.from_file(
                str(path),
                shared=False,
                size=expected["tokens"],
                dtype=torch.int16,
            )

    def check_length(self, seq_len: int) -> None:
        for split, stream in self.streams.items():
            if stream.numel() < seq_len + 1:
                raise ValueError(
                    f"{split} has {stream.numel()} tokens; need at least {seq_len + 1}. "
                    "Use more text or a smaller --seq-len."
                )

    def batch(
        self,
        split: Split,
        batch_size: int,
        seq_len: int,
        generator: torch.Generator,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stream = self.streams[split]
        starts = torch.randint(stream.numel() - seq_len, (batch_size,), generator=generator)
        indices = starts[:, None] + torch.arange(seq_len + 1)[None, :]
        chunk = stream[indices].to(device=device, dtype=torch.long)
        return chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
