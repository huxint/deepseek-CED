"""Fixed UTF-8 byte vocabulary; no downloaded tokenizer or unknown characters."""


class ByteTokenizer:
    vocab_size = 258
    bos_id = 256
    eos_id = 257
    name = "utf8-bytes-v1"

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        return (
            ([self.bos_id] if bos else [])
            + list(text.encode("utf-8"))
            + ([self.eos_id] if eos else [])
        )

    def decode(self, tokens: list[int]) -> str:
        return bytes(t for t in tokens if t not in (self.bos_id, self.eos_id)).decode(
            "utf-8",
            errors="replace",
        )
