import dataclasses
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, TypedDict, Union

from typing_extensions import Unpack

from outlines.generate.api import GenerationParameters, SamplingParameters
from outlines.models.transformers import TransformerTokenizer

if TYPE_CHECKING:
    from exllamav2 import ExLlamaV2Lora
    from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler


class ExllamaV2Params(TypedDict, total=False):
    max_tokens: int
    stop_conditions: Optional[List[Union[int, str]]]
    seed: Optional[int]
    gen_settings: "ExLlamaV2Sampler.Settings"
    max_new_tokens: List[int]


class ExLlamaV2Model:
    """Represents a `exl2` model."""

    def __init__(
        self,
        generator: "ExLlamaV2DynamicGenerator",
        tokenizer: TransformerTokenizer,
        max_seq_len: int,
    ):
        self.generator = generator
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def prepare_generation_parameters(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        sampling_parameters: SamplingParameters,
        structure_logits_processor,
        **exllamav2_params: Unpack[ExllamaV2Params],
    ) -> Tuple[ExllamaV2Params, Union[str, List[str]]]:
        """Prepare the generation parameters.

        `exllamav2` uses different default values

        """
        try:
            from exllamav2.generator import ExLlamaV2Sampler
        except ImportError:
            raise ImportError(
                "The `exllamav2` and `torch` libraries needs to be installed in order to use `exllamav2` models."
            )
        if isinstance(prompts, str):
            prompts = [prompts]
        max_tokens, stop_at, seed = dataclasses.astuple(generation_parameters)

        if max_tokens is None:
            max_tokens = []
            for prompt in prompts:
                ids = self.generator.tokenizer.encode(
                    prompt, encode_special_tokens=False
                )
                prompt_tokens = ids.shape[-1]
                max_tokens.append(self.max_seq_len - prompt_tokens)
            exllamav2_params["max_new_tokens"] = max_tokens
        else:
            exllamav2_params["max_new_tokens"] = [
                max_tokens for _ in range(len(prompts))
            ]

        stop_conditions = [self.generator.tokenizer.eos_token_id]
        if isinstance(generation_parameters.stop_at, str):
            stop_conditions.append(generation_parameters.stop_at)
        elif isinstance(generation_parameters.stop_at, list):
            for stop_at in generation_parameters.stop_at:
                stop_conditions.append(stop_at)
        exllamav2_params["stop_conditions"] = stop_conditions
        exllamav2_params["seed"] = seed

        gen_settings = ExLlamaV2Sampler.Settings()
        if sampling_parameters.temperature is not None:
            gen_settings.temperature = sampling_parameters.temperature
        if sampling_parameters.top_p is not None:
            gen_settings.top_p = sampling_parameters.top_p
        if sampling_parameters.top_k is not None:
            gen_settings.top_k = sampling_parameters.top_k
        gen_settings.logits_processor = structure_logits_processor
        exllamav2_params["gen_settings"] = gen_settings
        if sampling_parameters.num_samples > 1:
            prompts = prompts * sampling_parameters.num_samples

        if len(prompts) == 1:
            prompts = prompts[0]

        return exllamav2_params, prompts

    def reformat_output(
        self, output: Union[str, List[str]], sampling_parameters: SamplingParameters
    ):
        if isinstance(output, str):
            return output
        if len(output) == 1:
            return output[0]
        if sampling_parameters.num_samples > 1:
            if len(output) == sampling_parameters.num_samples:
                return output
            assert len(output) % sampling_parameters.num_samples == 0
            num_items_per_sample = len(output) // sampling_parameters.num_samples
            new_output = []
            for i in range(sampling_parameters.num_samples):
                curr_sample = []
                for j in range(num_items_per_sample):
                    curr_sample.append(output[i * num_items_per_sample + j])
                new_output.append(curr_sample)
            return new_output
        return output

    def generate(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        structure_logits_processor,
        sampling_parameters: SamplingParameters,
        **exllamav2_params: Unpack[ExllamaV2Params],
    ) -> Union[str, List[str]]:
        exllamav2_params, prompts = self.prepare_generation_parameters(
            prompts,
            generation_parameters,
            sampling_parameters,
            structure_logits_processor,
        )
        output = self.generator.generate(
            prompt=prompts,
            gen_settings=exllamav2_params["gen_settings"],
            max_new_tokens=min(exllamav2_params["max_new_tokens"]),
            completion_only=True,
            encode_special_tokens=False,
            stop_conditions=exllamav2_params["stop_conditions"],
            add_bos=False,
            seed=exllamav2_params["seed"],
        )

        return self.reformat_output(output, sampling_parameters)

    def stream(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        structure_logits_processor,
        sampling_parameters: SamplingParameters,
        **exllamav2_params: Unpack[ExllamaV2Params],
    ) -> Iterator[Union[str, List[str]]]:
        try:
            from exllamav2.generator import ExLlamaV2DynamicJob
        except ImportError:
            raise ImportError(
                "The `exllamav2` and `torch` libraries needs to be installed in order to use `exllamav2` models."
            )
        exllamav2_params, prompts = self.prepare_generation_parameters(
            prompts,
            generation_parameters,
            sampling_parameters,
            structure_logits_processor,
        )

        order = {}
        if isinstance(prompts, str):
            prompts = [prompts]
        batch_size = len(prompts)
        seed = exllamav2_params["seed"]
        for idx, p in enumerate(prompts):
            input_ids = self.generator.tokenizer.encode(
                p, encode_special_tokens=False, add_bos=False
            )

            job = ExLlamaV2DynamicJob(
                input_ids=input_ids,
                max_new_tokens=exllamav2_params["max_new_tokens"][idx],
                min_new_tokens=0,
                seed=seed,
                stop_conditions=exllamav2_params["stop_conditions"],
                gen_settings=exllamav2_params["gen_settings"],
                token_healing=False,
                decode_special_tokens=False,
            )

            if seed is not None:
                seed += 1

            serial = self.generator.enqueue(job)
            order[serial] = idx

        # Collect outputs until all jobs finish

        next_text = [""] * batch_size

        def token_generator() -> Iterator[str]:
            while self.generator.num_remaining_jobs():
                results = self.generator.iterate()
                for r in results:
                    idx = order[r["serial"]]
                    if r["stage"] == "streaming":
                        text = r.get("text", "")
                        next_text[idx] = text
                    if r["eos"]:
                        next_text[idx] = ""
                yield self.reformat_output(next_text, sampling_parameters)
            return

        return token_generator()

    def load_lora(self, adapter_path: str):
        try:
            from exllamav2 import ExLlamaV2Lora
        except ImportError:
            raise ImportError(
                "The `exllamav2` and `torch` libraries needs to be installed in order to use `exllamav2` models."
            )
        loras = [ExLlamaV2Lora.from_directory(self.generator.model, adapter_path)]
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
    lora: Optional["ExLlamaV2Lora"] = None,
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
            ExLlamaV2Tokenizer,
        )
        from exllamav2.generator import ExLlamaV2DynamicGenerator
        from transformers import AutoTokenizer

    except ImportError:
        raise ImportError(
            "The `exllamav2` and `torch` libraries needs to be installed in order to use `exllamav2` models."
        )
    config = ExLlamaV2Config(model_path)
    if max_chunk_size is not None:
        config.max_input_len = max_chunk_size
        config.max_attention_size = max_chunk_size**2

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
            draft_cache = ExLlamaV2Cache_8bit(
                draft_model, max_seq_len=max_seq_len, lazy=True
            )
        elif cache_q4:
            draft_cache = ExLlamaV2Cache_Q4(
                draft_model, max_seq_len=max_seq_len, lazy=True
            )
        else:
            draft_cache = ExLlamaV2Cache(
                draft_model, max_seq_len=max_seq_len, lazy=True
            )

    # Initialize the generator with all default parameters
    generator = ExLlamaV2DynamicGenerator(
        model=model,
        cache=cache,
        draft_model=draft_model,
        draft_cache=draft_cache,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
        use_ngram_draft=False,
        max_chunk_size=max_chunk_size,
        paged=paged,
    )
    if lora is not None:
        generator.set_loras(lora)
    hf_tokenizer_kwargs: dict[str, Any] = {}
    hf_tokenizer_kwargs.setdefault("padding_side", "left")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path, **hf_tokenizer_kwargs)
    max_seq_len = cache.max_seq_len
    return ExLlamaV2Model(generator, TransformerTokenizer(hf_tokenizer), max_seq_len)
