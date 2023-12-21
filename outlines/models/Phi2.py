import argparse
from typing import Optional
from dataclasses import dataclass
from mlx.utils import tree_unflatten
from transformers import AutoTokenizer

import mlx.core as mx
import mlx.nn as nn
import math
from pathlib import Path

@dataclass
class ModelArgs:
    max_sequence_length: int = 2048
    num_vocab: int = 51200
    model_dim: int = 2560
    num_heads: int = 32
    num_layers: int = 32
    rotary_dim: int = 32


class LayerNorm(nn.LayerNorm):
    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.astype(mx.float32)).astype(x.dtype)


class RoPEAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int, rotary_dim: int):
        super().__init__()

        self.num_heads = num_heads

        self.rope = nn.RoPE(rotary_dim, traditional=False)
        self.Wqkv = nn.Linear(dims, 3 * dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, x, mask=None, cache=None):
        qkv = self.Wqkv(x)
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        # Extract some shapes
        num_heads = self.num_heads
        B, L, D = queries.shape

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)

        # Add RoPE to the queries and keys and combine them with the cache
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        queries = queries.astype(mx.float32)
        keys = keys.astype(mx.float32)

        # Finally perform the attention computation
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask

        scores = mx.softmax(scores, axis=-1).astype(values.dtype)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.out_proj(values_hat), (keys, values)


class ParallelBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        dims = config.model_dim
        mlp_dims = dims * 4
        self.mixer = RoPEAttention(dims, config.num_heads, config.rotary_dim)
        self.ln = LayerNorm(dims)
        self.fc1 = nn.Linear(dims, mlp_dims)
        self.fc2 = nn.Linear(mlp_dims, dims)
        self.act = nn.GELU(approx="precise")

    def __call__(self, x, mask, cache):
        h = self.ln(x)
        attn_h, cache = self.mixer(h, mask, cache)
        ff_h = self.fc2(self.act(self.fc1(h)))
        return attn_h + ff_h + x, cache


class TransformerDecoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.h = [ParallelBlock(config) for i in range(config.num_layers)]

    def __call__(self, x, mask, cache):
        if cache is None:
            cache = [None] * len(self.h)

        for e, layer in enumerate(self.h):
            x, cache[e] = layer(x, mask, cache[e])
        return x, cache


class OutputHead(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        self.ln = LayerNorm(config.model_dim)
        self.linear = nn.Linear(config.model_dim, config.num_vocab)

    def __call__(self, inputs):
        return self.linear(self.ln(inputs))


class Phi2(nn.Module):
    def __init__(self, config: ModelArgs):
        self.wte = nn.Embedding(config.num_vocab, config.model_dim)
        self.transformer = TransformerDecoder(config)
        self.lm_head = OutputHead(config)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache: mx.array = None,
    ) -> tuple[mx.array, mx.array]:
        

        x = self.wte(inputs)

        mask = None
        if x.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            mask = mask.astype(x.dtype)

        y, cache = self.transformer(x, mask, cache)
        
        return self.lm_head(y), cache

def load_model(model_name: str):

    model = Phi2(ModelArgs())
    model_path = "/Users/sachaichbiah/Documents/CARGOHUB/mlx-examples/phi2/"
    model_path = Path(model_path)
    weights = mx.load(str(model_path / "weights.npz"))
    model.update(tree_unflatten(list(weights.items())))

    return model