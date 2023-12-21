# Copyright © 2023 Apple Inc.

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer

@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float
    rope_traditional: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output


class RoPE(nn.RoPE):
    def __init__(self, dims: int, traditional: bool = False, base: float = 10000):
        super().__init__(dims, traditional)
        self.base = base

    def __call__(self, x, offset: int = 0):
        shape = x.shape
        x = mx.reshape(x, (-1, shape[-2], shape[-1]))
        N = x.shape[1] + offset
        costheta, sintheta = RoPE.create_cos_sin_theta(
            N, self.dims, offset=offset, base=self.base, dtype=x.dtype
        )

        rope = (
            self._compute_traditional_rope if self.traditional else self._compute_rope
        )
        rx = rope(costheta, sintheta, x)

        return mx.reshape(rx, shape)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.rope = RoPE(
            args.head_dim, traditional=args.rope_traditional, base=args.rope_theta
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.n_heads, L, -1])

        keys, values = map(repeat, (keys, values))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output), (keys, values)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache

"""
class Llama(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(self, 
                 inputs: mx.array,
                 mask: mx.array = None,
                 cache: mx.array = None,
                 ) -> tuple[mx.array, mx.array]:
        
        mask = None
        if x.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            mask = mask.astype(self.tok_embeddings.weight.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            x, cache[e] = layer(x, mask, cache[e])

        for l in self.layers:
            x, cache[i] = l(x, mask, cache=cache[i])

        = self.layers[i](x, mask=None, 
        x = self.norm(x)
        return self.output(x)"""
class Llama(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(self, 
                 inputs: mx.array,
                 mask: mx.array = None,
                 cache: mx.array = None,
                 ) -> tuple[mx.array, mx.array]:
        
        x = self.tok_embeddings(inputs)
        mask = None
        if x.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            mask = mask.astype(self.tok_embeddings.weight.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            x, cache[e] = layer(x, mask, cache[e])
        x = self.norm(x)
        return self.output(x), cache

    def generate(self, x, temp=1.0):
        cache = []

        # Make an additive causal mask. We will need that to process the prompt.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.tok_embeddings.weight.dtype)

        # First we process the prompt x the same was as in __call__ but
        # save the caches in cache
        x = self.tok_embeddings(x)
        for l in self.layers:
            x, c = l(x, mask=mask)
            # We store the per layer cache in a simple python list
            cache.append(c)
        x = self.norm(x)
        # We only care about the last logits that generate the next token
        y = self.output(x[:, -1])
        y = mx.random.categorical(y * (1 / temp))

        # y now has size [1]
        # Since MLX is lazily evaluated nothing is computed yet.
        # Calling y.item() would force the computation to happen at
        # this point but we can also choose not to do that and let the
        # user choose when to start the computation.
        yield y

        # Now we parsed the prompt and generated the first token we
        # need to feed it back into the model and loop to generate the
        # rest.
        while True:
            # Unsqueezing the last dimension to add a sequence length
            # dimension of 1
            x = y[:, None]

            x = self.tok_embeddings(x)
            for i in range(len(cache)):
                # We are overwriting the arrays in the cache list. When
                # the computation will happen, MLX will be discarding the
                # old cache the moment it is not needed anymore.
                x, cache[i] = self.layers[i](x, mask=None, cache=cache[i])
            x = self.norm(x)
            y = self.output(x[:, -1])
            y = mx.random.categorical(y * (1 / temp))

            yield y


def tic():
    return time.time()


def toc(msg, start):
    end = time.time()
    return f"[INFO] {msg}: {end - start:.3f} s"


def load_model(model_name:str):
    model_path_dict = {"TinyLlama/TinyLlama-1.1B-Chat-v0.6":"/Users/sachaichbiah/Documents/CARGOHUB/mlx-examples/llama/mlx_model"}
    model_path = model_path_dict[model_name]

    model_path = Path(model_path)
    weights = mx.load(str(model_path / "weights.npz"))
    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        n_heads = config["n_heads"]
        if "n_kv_heads" not in config:
            config["n_kv_heads"] = n_heads
        if "head_dim" not in config:
            config["head_dim"] = config["dim"] // n_heads
        if "hidden_dim" not in config:
            config["hidden_dim"] = weights["layers.0.feed_forward.w1.weight"].shape[0]
        if config.get("vocab_size", -1) < 0:
            config["vocab_size"] = weights["output.weight"].shape[-1]
        if "rope_theta" not in config:
            config["rope_theta"] = 10000
        unused = ["multiple_of", "ffn_dim_multiplier"]
        for k in unused:
            if k in config:
                config.pop(k)
    model = Llama(ModelArgs(**config))
    model.update(tree_unflatten(list(weights.items())))
    return model
