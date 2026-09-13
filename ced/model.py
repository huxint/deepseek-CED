"""Dense CED core, based on DeepSeek-V4.1-Flash report sections 2.2 and 3.2.2.

The decoder gets global K/V exclusively from the final causal encoder output.
Every layer also has its own sliding-window K/V. Local and global entries share
one attention softmax. This is an ordinary GQA analogue of the paper's latent,
compressed sparse attention, not a reproduction of CSA2 or the released weights.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig

type PrefillMode = Literal["exact", "bounded"]


@dataclass(frozen=True, slots=True)
class KVCache:
    # [batch, kv_heads, time, head_dim], already position-rotated keys.
    key: Tensor
    value: Tensor
    start: int

    @property
    def end(self) -> int:
        return self.start + self.key.size(2)

    def append(self, other: KVCache) -> KVCache:
        return KVCache(
            torch.cat((self.key, other.key), dim=2),
            torch.cat((self.value, other.value), dim=2),
            self.start,
        )

    def tail(self, size: int) -> KVCache:
        length = min(size, self.key.size(2))
        # Clone, not a view: otherwise a small SWA cache keeps the full prompt allocation alive.
        return KVCache(
            self.key[:, :, -length:].clone(),
            self.value[:, :, -length:].clone(),
            self.end - length,
        )


@dataclass(frozen=True, slots=True)
class LayerCache:
    local: KVCache
    global_kv: KVCache | None = None


type LayerCaches = tuple[LayerCache, ...]
type MemoryBanks = tuple[KVCache, ...]


@dataclass(frozen=True, slots=True)
class CEDCache:
    encoder: LayerCaches
    decoder: LayerCaches
    memory: MemoryBanks  # One allocation per decoder KV group, not per layer.
    length: int
    approximate: bool
    decoder_prefill_tokens: int


@dataclass(frozen=True, slots=True)
class LMOutput:
    logits: Tensor
    loss: Tensor | None = None


@dataclass(frozen=True, slots=True)
class CachedOutput:
    logits: Tensor
    cache: CEDCache


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class Rotary(nn.Module):
    inv_freq: Tensor

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim),
            persistent=False,
        )

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        angles = positions.float()[:, None] * self.inv_freq.float()[None, :]
        cos = angles.cos().to(x.dtype)[None, None]
        sin = angles.sin().to(x.dtype)[None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class KVProjection(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.linear = nn.Linear(config.d_model, 2 * self.heads * self.head_dim, bias=False)
        self.rope = Rotary(self.head_dim, config.rope_theta)

    def forward(self, x: Tensor, positions: Tensor, start: int) -> KVCache:
        key, value = self.linear(x).chunk(2, dim=-1)
        shape = (*x.shape[:2], self.heads, self.head_dim)
        key = self.rope(key.reshape(shape).transpose(1, 2), positions)
        value = value.reshape(shape).transpose(1, 2)
        return KVCache(key, value, start)


class EncoderMemory(nn.Module):
    """A decoder global-KV source. It never receives a decoder hidden state."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.projection = KVProjection(config)

    def forward(self, encoder_hidden: Tensor, positions: Tensor, start: int) -> KVCache:
        return self.projection(self.norm(encoder_hidden), positions, start)


class HybridAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.kv_repeat = config.n_heads // config.n_kv_heads
        self.window = config.window_size
        self.dropout = config.dropout
        self.query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.local_kv = KVProjection(config)
        self.rope = Rotary(self.head_dim, config.rope_theta)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        start: int,
        global_kv: KVCache | None,
        past: KVCache | None,
        use_cache: bool,
    ) -> tuple[Tensor, KVCache | None]:
        batch, length, width = x.shape
        query = self.query(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        query = self.rope(query, positions)
        local = self.local_kv(x, positions, start)
        if past is not None:
            local = past.append(local)
        local_positions = torch.arange(local.start, local.end, device=x.device)
        mask = (local_positions[None, :] <= positions[:, None]) & (
            local_positions[None, :] >= positions[:, None] - self.window + 1
        )
        key, value = local.key, local.value
        if global_kv is not None:
            global_positions = torch.arange(global_kv.start, global_kv.end, device=x.device)
            # Essential even for encoder-derived memory: future encoder tokens contain the label.
            global_mask = global_positions[None, :] <= positions[:, None]
            key = torch.cat((key, global_kv.key), dim=2)
            value = torch.cat((value, global_kv.value), dim=2)
            mask = torch.cat((mask, global_mask), dim=-1)
        if self.kv_repeat != 1:
            key = key.repeat_interleave(self.kv_repeat, dim=1)
            value = value.repeat_interleave(self.kv_repeat, dim=1)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(attended), local.tail(self.window) if use_cache else None


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.up = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig, own_global_kv: bool = False) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = HybridAttention(config)
        self.global_projection = KVProjection(config) if own_global_kv else None
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config)
        self.dropout = config.dropout

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        start: int,
        memory: KVCache | None = None,
        past: LayerCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, LayerCache | None]:
        normalized = self.attn_norm(x)
        if self.global_projection is not None:
            projected: KVCache = self.global_projection(normalized, positions, start)
            memory = (
                past.global_kv.append(projected)
                if past is not None and past.global_kv is not None
                else projected
            )
        attention, local = self.attention(
            normalized,
            positions,
            start,
            memory,
            past.local if past else None,
            use_cache,
        )
        x = x + F.dropout(attention, self.dropout, self.training)
        x = x + F.dropout(self.ffn(self.ffn_norm(x)), self.dropout, self.training)
        cache = None
        if local is not None:
            cache = LayerCache(local, memory if self.global_projection is not None else None)
        return x, cache


class CEDLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        encoder_blocks = [
            Block(config, own_global_kv=i >= config.encoder_swa_only_layers)
            for i in range(config.encoder_layers)
        ]
        decoder_blocks = [Block(config) for _ in range(config.decoder_layers)]
        self.encoder = nn.ModuleList(encoder_blocks)
        self.memory_banks = nn.ModuleList(
            EncoderMemory(config) for _ in range(config.decoder_kv_groups)
        )
        self.decoder = nn.ModuleList(decoder_blocks)
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        residual_std = 0.02 / math.sqrt(2 * (config.encoder_layers + config.decoder_layers))
        for block in (*encoder_blocks, *decoder_blocks):
            nn.init.normal_(block.attention.output.weight, std=residual_std)
            nn.init.normal_(block.ffn.down.weight, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _positions(self, tokens: Tensor, start: int = 0) -> Tensor:
        if start + tokens.size(1) > self.config.max_seq_len:
            raise ValueError(f"sequence exceeds max_seq_len={self.config.max_seq_len}")
        return torch.arange(start, start + tokens.size(1), device=tokens.device)

    def _encode(
        self,
        tokens: Tensor,
        positions: Tensor,
        start: int,
        past: LayerCaches | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, LayerCaches]:
        hidden = self.embedding(tokens)
        caches: list[LayerCache] = []
        for index, block in enumerate(self.encoder):
            hidden, cache = block(
                hidden,
                positions,
                start,
                past=past[index] if past else None,
                use_cache=use_cache,
            )
            if cache is not None:
                caches.append(cache)
        return hidden, tuple(caches)

    def _memory(
        self,
        encoded: Tensor,
        positions: Tensor,
        start: int,
        past: MemoryBanks | None = None,
    ) -> MemoryBanks:
        banks: list[KVCache] = []
        for index, bank in enumerate(self.memory_banks):
            kv = bank(encoded, positions, start)
            banks.append(past[index].append(kv) if past else kv)
        return tuple(banks)

    def _decode(
        self,
        hidden: Tensor,
        positions: Tensor,
        start: int,
        memory: MemoryBanks,
        past: LayerCaches | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, LayerCaches]:
        caches: list[LayerCache] = []
        layers_per_group = self.config.decoder_layers // self.config.decoder_kv_groups
        for index, block in enumerate(self.decoder):
            hidden, cache = block(
                hidden,
                positions,
                start,
                memory=memory[index // layers_per_group],
                past=past[index] if past else None,
                use_cache=use_cache,
            )
            if cache is not None:
                caches.append(cache)
        return hidden, tuple(caches)

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> LMOutput:
        """Teacher forcing. targets[b, t] must be the token AFTER input_ids[b, t]."""
        positions = self._positions(input_ids)
        encoded, _ = self._encode(input_ids, positions, 0)
        memory = self._memory(encoded, positions, 0)
        hidden, _ = self._decode(encoded, positions, 0, memory)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100
            )
        return LMOutput(logits, loss)

    @torch.inference_mode()
    def prefill(self, input_ids: Tensor, mode: PrefillMode = "exact") -> CachedOutput:
        """Return last-token logits [B, 1, V] and caches. Bounded replay is approximate."""
        positions = self._positions(input_ids)
        encoded, encoder = self._encode(input_ids, positions, 0, use_cache=True)
        memory = self._memory(encoded, positions, 0)
        length = input_ids.size(1)
        start = max(0, length - self.config.window_size) if mode == "bounded" else 0
        hidden, decoder = self._decode(
            encoded[:, start:],
            positions[start:],
            start,
            memory,
            use_cache=True,
        )
        cache = CEDCache(encoder, decoder, memory, length, start > 0, length - start)
        return CachedOutput(self.lm_head(self.final_norm(hidden[:, -1:])), cache)

    @torch.inference_mode()
    def decode(self, input_ids: Tensor, cache: CEDCache) -> CachedOutput:
        """Append one token or a contiguous chunk; return logits [B, new_time, V]."""
        start = cache.length
        positions = self._positions(input_ids, start)
        encoded, encoder = self._encode(input_ids, positions, start, cache.encoder, True)
        memory = self._memory(encoded, positions, start, cache.memory)
        hidden, decoder = self._decode(encoded, positions, start, memory, cache.decoder, True)
        updated = CEDCache(
            encoder,
            decoder,
            memory,
            start + input_ids.size(1),
            cache.approximate,
            cache.decoder_prefill_tokens,
        )
        return CachedOutput(self.lm_head(self.final_norm(hidden)), updated)
