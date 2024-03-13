import os
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Lora
    from transformers import PreTrainedTokenizer

from .transformers import TransformerTokenizer


class ExLlamaV2Model:
    """Represents a `exl2` model."""

    def __init__(
        self,
        model: "ExLlamaV2",
        tokenizer: "PreTrainedTokenizer",
        device,
        cache: "ExLlamaV2Cache",
        lora: Optional["ExLlamaV2Lora"] = None,
    ):
        self.device = device
        self.model = model
        self.tokenizer = TransformerTokenizer(tokenizer)
        self.cache = cache
        self.past_seq = None
        self.lora = lora

    def forward(self, input_ids: torch.LongTensor, *_):
        """Compute a forward pass through the exl2 model."""

        # Caching with past_seq
        reset = True
        seq_tensor = input_ids[0]

        if self.past_seq is not None:
            min_length = min(self.past_seq.shape[0], seq_tensor.shape[0])
            indices = torch.nonzero(
                ~torch.eq(self.past_seq[:min_length], seq_tensor[:min_length])
            )
            if len(indices) > 0:
                longest_prefix = indices[0].item()
            else:
                longest_prefix = min_length

            if longest_prefix > 0:
                reset = False
                self.cache.current_seq_len = longest_prefix
                if seq_tensor.shape[0] - longest_prefix > 1:
                    self.model.forward(
                        seq_tensor[longest_prefix:-1].view(1, -1),
                        self.cache,
                        preprocess_only=True,
                        loras=[self.lora],
                    )
                elif seq_tensor.shape[0] == longest_prefix:
                    self.cache.current_seq_len -= 1

        if reset:
            self.cache.current_seq_len = 0
            if seq_tensor.shape[0] > 1:
                self.model.forward(
                    seq_tensor[:-1].view(1, -1),
                    self.cache,
                    preprocess_only=True,
                    loras=[self.lora],
                )

        self.past_seq = seq_tensor

        return self.model.forward(
            seq_tensor[-1:].view(1, -1), self.cache, loras=[self.lora]
        )

    def __call__(self, input_ids: torch.LongTensor, *_) -> torch.FloatTensor:
        logits = self.forward(input_ids)
        next_token_logits = logits[..., -1, :]

        return next_token_logits, None

    def update_lora(self, lora_path: Optional[str] = None):
        """
        Update and apply the LoRA to the model.

        Args:
            lora_path (Optional[str]): The path to the LoRA directory. If None, the LoRA will be unloaded.
        """
        try:
            from exllamav2 import ExLlamaV2Lora
        except ImportError:
            raise ImportError(
                "The `exllamav2` library needs to be installed in order to use `exllamav2` models."
            )
        if lora_path is None:
            if self.lora is not None:
                print(" -- Unloading LoRA...")
            self.lora = None
        else:
            self.lora = ExLlamaV2Lora.from_directory(self.model, lora_path)
            print(" -- Loading LoRA...")


def exl2(
    model_path: str,
    device: str,
    max_seq_len: Optional[int] = None,
    scale_pos_emb: Optional[float] = None,
    scale_alpha_value: Optional[float] = None,
    no_flash_attn: Optional[bool] = None,
    num_experts_per_token: Optional[int] = None,
    cache_8bit: bool = False,
    cache_q4: bool = False,
    tokenizer_kwargs: dict = {},
    gpu_split: Optional[str] = None,
    low_mem: Optional[bool] = None,
    verbose: Optional[bool] = None,
) -> ExLlamaV2Model:
    """
    Load an ExLlamaV2 model.

    Parameters
    ----------
    model_path (str)
        Path to the model directory.
    device (str)
        Device to load the model on. Pass in 'cuda' for GPU or 'cpu' for CPU
    max_seq_len (Optional[int], optional)
        Maximum sequence length. Defaults to None.
    scale_pos_emb (Optional[float], optional)
        Scale factor for positional embeddings. Defaults to None.
    scale_alpha_value (Optional[float], optional)
        Scale alpha value. Defaults to None.
    no_flash_attn (Optional[bool], optional)
        Disable flash attention. Defaults to None.
    num_experts_per_token (Optional[int], optional)
        Number of experts per token. Defaults to None.
    cache_8bit (bool, optional)
        Use 8-bit cache. Defaults to False.
    cache_q4 (bool, optional)
        Use Q4 cache. Defaults to False.
    tokenizer_kwargs (dict, optional)
        Additional keyword arguments for the tokenizer. Defaults to {}.
    gpu_split (str)
        \"auto\", or VRAM allocation per GPU in GB. Auto will use exllama's autosplit feature
    low_mem (bool, optional)
        Enable VRAM optimizations, potentially trading off speed
    verbose (bool, optional)
        Enable if you want debugging statements

    Returns
    -------
    An `ExLlamaV2Model` instance.

    Raises
    ------
    `ImportError` if the `exllamav2` library is not installed.

    """
    try:
        from exllamav2 import (
            ExLlamaV2,
            ExLlamaV2Cache,
            ExLlamaV2Cache_8bit,
            ExLlamaV2Cache_Q4,
            ExLlamaV2Config,
        )
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "The `exllamav2` library needs to be installed in order to use `exllamav2` models."
        )

    # Load tokenizer
    if not verbose:
        print(" -- Loading tokenizer...")
    tokenizer_kwargs.setdefault("padding_side", "left")
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    # tokenizer = TransformerTokenizer(model_path, **tokenizer_kwargs)

    # Check fasttensors for config
    if os.name != "nt":
        use_fasttensors = True
    else:
        use_fasttensors = False

    # Create config
    config = ExLlamaV2Config()
    config.model_dir = model_path
    config.fasttensors = use_fasttensors
    config.prepare()

    # Set config options
    if max_seq_len is not None:
        config.max_seq_len = max_seq_len
    if scale_pos_emb is not None:
        config.scale_pos_emb = scale_pos_emb
    if scale_alpha_value is not None:
        config.scale_alpha_value = scale_alpha_value
    if no_flash_attn is not None:
        config.no_flash_attn = no_flash_attn
    if num_experts_per_token is not None:
        config.num_experts_per_token = num_experts_per_token
    if low_mem:
        config.set_low_mem()

    # Prepare the model from the config
    model = ExLlamaV2(config)

    # Create cache
    if cache_8bit:
        cache = ExLlamaV2Cache_8bit(model, lazy=not model.loaded)
    elif cache_q4:
        cache = ExLlamaV2Cache_Q4(model, lazy=not model.loaded)
    else:
        cache = ExLlamaV2Cache(model, lazy=not model.loaded)

    # Load the model
    split = None
    if gpu_split and gpu_split != "auto":
        split = [float(alloc) for alloc in gpu_split.split(",")]
        if not verbose:
            print(" -- Loading model...")
        model.load(split)

    # Autoload if no GPU split was provided
    if not model.loaded:
        print(" -- Loading model...")
        model.load_autosplit(cache)

    return ExLlamaV2Model(model, tokenizer, device, cache)
