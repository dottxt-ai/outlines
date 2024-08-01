import os
from typing import (
    TYPE_CHECKING,
    Iterator,
    List,
    Optional,
    TypedDict,
    Union,
)

from typing_extensions import Unpack

from outlines.generate.api import GenerationParameters, SamplingParameters

if TYPE_CHECKING:
    from exllamav2 import ExLlamaV2, ExLlamaV2Lora, ExLlamaV2Tokenizer
    from exllamav2.generator import ExLlamaV2Sampler
    from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler, ExLlamaV2DynamicJob


class ExllamaV2Params(TypedDict, total=False):
    min_new_tokens: int = 0,
    seed: Union[int, None] = None,
    encode_special_tokens: bool = False,
    decode_special_tokens: bool = False,
    add_bos: bool = False,

class ExLlamaV2Model:
    """Represents a `exl2` model."""
    def __init__(
        self,
        generator: ExLlamaV2DynamicGenerator,
    ):
        self.generator = generator
    def generate(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        structure_logits_processor,
        sampling_parameters: SamplingParameters,
        **exllamav2_params: Unpack[ExllamaV2Params]
    ) -> List[str]:
        stop_conditions = [self.generator.tokenizer.eos_token_id]
        if isinstance(generation_parameters.stop_at, str):
            stop_conditions.append(generation_parameters.stop_at)
        elif isinstance(generation_parameters.stop_at, list):
            for stop_at in generation_parameters.stop_at:
                stop_conditions.append(stop_at)
        gen_settings = ExLlamaV2Sampler.Settings()
        if sampling_parameters.temperature is not None:
            gen_settings.temperature = sampling_parameters.temperature
        if sampling_parameters.top_p is not None:
            gen_settings.top_p = sampling_parameters.top_p
        if sampling_parameters.top_k is not None:
            gen_settings.top_k = sampling_parameters.top_k
        gen_settings.logits_processor = structure_logits_processor
        return self.generator.generate(
            prompt=prompts,
            gen_settings=gen_settings,
            max_new_tokens=generation_parameters.max_tokens,
            completion_only=True,
            encode_special_tokens=exllamav2_params.encode_special_tokens,
            stop_conditions=stop_conditions,
            add_bos=exllamav2_params.add_bos,
            seed=generation_parameters.seed,
        )
    def stream(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        structure_logits_processor,
        sampling_parameters: SamplingParameters,
        **exllamav2_params: Unpack[ExllamaV2Params]
    ) -> List[Iterator[str]]:
        gen_settings = ExLlamaV2Sampler.Settings()
        if sampling_parameters.temperature is not None:
            gen_settings.temperature = sampling_parameters.temperature
        if sampling_parameters.top_p is not None:
            gen_settings.top_p = sampling_parameters.top_p
        if sampling_parameters.top_k is not None:
            gen_settings.top_k = sampling_parameters.top_k
        gen_settings.logits_processor = structure_logits_processor
        stop_conditions = [self.generator.tokenizer.eos_token_id]
        if isinstance(generation_parameters.stop_at, str):
            stop_conditions.append(generation_parameters.stop_at)
        elif isinstance(generation_parameters.stop_at, list):
            for stop_at in generation_parameters.stop_at:
                stop_conditions.append(stop_at)
        order = {}
        if isinstance(prompts, str):
            prompts = [prompts]
        seed = generation_parameters.seed
        batch_size = len(prompts)
        for idx, p in enumerate(prompts):

            input_ids = self.generator.tokenizer.encode(p, encode_special_tokens = True, add_bos = False)

            job = ExLlamaV2DynamicJob(
                input_ids = input_ids,
                max_new_tokens = generation_parameters.max_tokens,
                min_new_tokens = exllamav2_params.min_new_tokens,
                seed = exllamav2_params.seed,
                stop_conditions = stop_conditions,
                gen_settings = gen_settings,
                token_healing = False,
                decode_special_tokens = exllamav2_params.decode_special_tokens,
            )

            if seed is not None: seed += 1

            serial = self.generator.enqueue(job)
            order[serial] = idx

        # Collect outputs until all jobs finish

        completions = [""] * batch_size

        def token_generator() -> Iterator[str]:
            while self.generator.num_remaining_jobs():
                results = self.generator.iterate()
                for r in results:
                    idx = order[r["serial"]]
                    if r["stage"] == "streaming":
                        all_eos = False
                        text = r.get("text", "")
                        completions[idx] += text
                    if r["eos"]:
                        completions[idx] = r
                yield completions
            return

        return token_generator()

    def load_lora(self, adapter_path: str):
        loras = [ExLlamaV2Lora.from_directory(self.model, adapter_path)]
        print(" -- Loading LoRA...")
        self.generator.set_loras(loras)


def exl2(
    model_path: str,
    draft_model_path: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    cache_8bit: bool = False,
    cache_q4: bool = False,
    paged: bool = True,
    max_chunk_size: Optional[int] = None,
    lora: Optional[ExLlamaV2Lora] = None

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
            "The `exllamav2`, `transformers` and `torch` libraries needs to be installed in order to use `exllamav2` models."
        )
    config = ExLlamaV2Config(model_path)
    if max_chunk_size is not None:
        config.max_input_len = max_chunk_size
        config.max_attention_size = max_chunk_size ** 2

    config.arch_compat_overrides()
    model = ExLlamaV2(config)
    if max_seq_len is None:
        max_seq_len = -1
    if cache_8bit:
        cache = ExLlamaV2Cache_8bit(model, max_seq_len=max_seq_len, lazy=True)
    elif cache_q4:
        cache = ExLlamaV2Cache_Q4(model, max_seq_len=max_seq_len, lazy=True)
    else:
        cache = ExLlamaV2Cache(model, max_seq_len=max_seq_len, lazy=True)
    model.load_autosplit(cache, progress=True)

    print("Loading tokenizer...")
    tokenizer = ExLlamaV2Tokenizer(config)
    tokenizer.vocabulary = tokenizer.extended_piece_to_id
    max_batch_size = 4 if paged else 1

    draft_model = None
    draft_cache = None
    if draft_model_path is not None:
        draft_config = ExLlamaV2Config(draft_model_path)
        draft_model = ExLlamaV2(draft_config)

        if cache_8bit:
            draft_cache = ExLlamaV2Cache_8bit(draft_model, max_seq_len=max_seq_len, lazy=True)
        elif cache_q4:
            draft_cache = ExLlamaV2Cache_Q4(draft_model, max_seq_len=max_seq_len, lazy=True)
        else:
            draft_cache = ExLlamaV2Cache(draft_model, max_seq_len=max_seq_len, lazy=True)


    # Initialize the generator with all default parameters
    generator = ExLlamaV2DynamicGenerator(
        model = model,
        cache = cache,
        draft_model = draft_model,
        draft_cache = draft_cache,
        tokenizer = tokenizer,
        max_batch_size = max_batch_size,
        use_ngram_draft = False,
        max_chunk_size = max_chunk_size,
        paged = paged,
    )
    if lora is not None:
        generator.set_loras(lora)
    return ExLlamaV2Model(generator)