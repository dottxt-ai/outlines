### Legacy API: file exists for backward compatibility with v0
### Do not use it as an example of how to write new code

import warnings
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union
)

if TYPE_CHECKING:
    import torch.LongTensor
    from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler  # type: ignore


def exllamav2_params_adapter(sampling_params: dict, generation_params: dict, kwargs: dict) -> dict:
    """Argument names mapping is done in the model class for the `exllamav2` model."""
    kwargs.update(**sampling_params, **generation_params)
    return kwargs


class ExllamaV2Params(TypedDict, total=False):
    stop_conditions: Optional[List[Union[int, str]]]
    seed: Optional[int]
    gen_settings: "ExLlamaV2Sampler.Settings"
    max_new_tokens: List[int]


class OutlinesExLlamaV2Tokenizer:
    def __init__(self, tokenizer):
        self.exl2_tokenizer = tokenizer
        self.vocabulary = self.exl2_tokenizer.get_piece_to_id_dict()
        self.special_tokens = set(self.exl2_tokenizer.extended_piece_to_id)
        self.eos_token_id = self.exl2_tokenizer.eos_token_id

    def convert_token_to_string(self, token):
        return token

    def decode(self, token_ids: "torch.LongTensor") -> List[str]:
        import torch

        decoded = self.exl2_tokenizer.decode(
            torch.tensor(token_ids),
            decode_special_tokens=False,
        )
        if isinstance(decoded, str):
            return [decoded]
        return decoded


class ExLlamaV2Model():
    """Represents a `exl2` model."""

    def __init__(
        self,
        generator: "ExLlamaV2DynamicGenerator",
        tokenizer: "OutlinesExLlamaV2Tokenizer",
        max_seq_len: int,
    ):
        warnings.warn("""
            The `ExLlamaV2Model` model is deprecated starting from v1.0.0.
            Support for it will be removed in v1.1.0.
            As the `exllamav2` library is not compatible with some key features
            of `outlines`, we decided to remove support for it.
            """,
            DeprecationWarning,
            stacklevel=2,
        )
        self.generator = generator
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "As a v0 model, the `exllamav2` model does not support the "
            + "`__call__` method."
        )

    def stream(self, *args, **kwargs):
        raise NotImplementedError(
            "As a v0 model, the `exllamav2` model does not support the "
            + "`stream` method."
        )

    def prepare_generation_parameters(
        self,
        prompts: Union[str, List[str]],
        logits_processor,
        kwargs
    ) -> Tuple[Dict, Union[str, List[str]]]:
        """Prepare the generation parameters.

        `exllamav2` uses different default values

        """
        from exllamav2.generator import ExLlamaV2Sampler

        exllamav2_params = {}

        if isinstance(prompts, str):
            prompts = [prompts]

        max_tokens = kwargs.get("max_tokens")
        if max_tokens is None:
            max_tokens = []
            for prompt in prompts:
                ids = self.generator.tokenizer.encode(
                    prompt, encode_special_tokens=True
                )
                prompt_tokens = ids.shape[-1]
                max_tokens.append(self.max_seq_len - prompt_tokens)
            exllamav2_params["max_new_tokens"] = max_tokens
        else:
            exllamav2_params["max_new_tokens"] = [
                max_tokens for _ in range(len(prompts))
            ]

        stop_at = kwargs.get("stop_at")
        stop_conditions = [self.generator.tokenizer.eos_token_id]
        if isinstance(stop_at, str):
            stop_conditions.append(stop_at)
        elif isinstance(stop_at, list):
            for stop_at in stop_at:
                stop_conditions.append(stop_at)
        exllamav2_params["stop_conditions"] = stop_conditions

        exllamav2_params["seed"] = kwargs.get("seed")

        gen_settings = ExLlamaV2Sampler.Settings()

        temperature = kwargs.get("temperature")
        if temperature is not None:
            gen_settings.temperature = temperature

        top_p = kwargs.get("top_p")
        if top_p is not None:
            gen_settings.top_p = top_p

        top_k = kwargs.get("top_k")
        if top_k is not None:
            gen_settings.top_k = top_k

        gen_settings.logits_processor = logits_processor

        exllamav2_params["gen_settings"] = gen_settings

        num_samples = kwargs.get("num_samples")
        if num_samples is not None and num_samples > 1:
            prompts = prompts * num_samples
            exllamav2_params["max_new_tokens"] = (
                exllamav2_params["max_new_tokens"] * num_samples
            )

        if len(prompts) == 1:
            prompts = prompts[0]

        return exllamav2_params, prompts

    def reformat_output(
        self, output: Union[str, List[str]], sampling_parameters
    ):
        """
        The purpose of this function is to reformat the output from exllamav2's output format to outline's output format.

        For exllamav2, it mainly accepts only a list or a string(they also do cfg sampling with tuples but we will ignore this for now).
        The exllamav2's logic is:

        1. If the prompt is a string, return a string. This is the same as outlines
        2. If a prompt is a list, return a list. This is not the same as outlines output in that if the list is only one element, the string is expected to be outputted.
        3. There is no such thing as num_samples, so the prompts had to be duplicated by num_samples times. Then, we had the function output a list of lists
        """
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
        logits_processor,
        **kwargs
    ) -> Union[str, List[str]]:
        exllamav2_params, prompts = self.prepare_generation_parameters(
            prompts,
            logits_processor,
            kwargs
        )
        """
        In exllamav2, it needs the max amount of new tokens generated.
        The reason exllamav2_params["max_new_tokens"] is a list is because in prepare_generation_parameters
        the max amount of tokens that can be generated by the model for each prompt(by encoding with tokenizer) is calculated.
        The minimum is picked because otherwise it might be possible for one of the
        prompts to exceed the max sequence length.
        """
        output = self.generator.generate(
            prompt=prompts,
            gen_settings=exllamav2_params["gen_settings"],
            max_new_tokens=min(exllamav2_params["max_new_tokens"]),
            completion_only=True,
            encode_special_tokens=True,
            stop_conditions=exllamav2_params["stop_conditions"],
            add_bos=False,
            seed=exllamav2_params["seed"],
        )

        return self.reformat_output(output, kwargs)

    def generate_stream(
        self,
        prompts: Union[str, List[str]],
        logits_processor,
        **kwargs
    ) -> Iterator[Union[str, List[str]]]:
        from exllamav2.generator import ExLlamaV2DynamicJob

        exllamav2_params, prompts = self.prepare_generation_parameters(
            prompts,
            logits_processor,
            kwargs
        )

        order = {}
        if isinstance(prompts, str):
            prompts = [prompts]
        batch_size = len(prompts)
        seed = exllamav2_params["seed"]
        for idx, p in enumerate(prompts):
            input_ids = self.generator.tokenizer.encode(
                p, encode_special_tokens=True, add_bos=False
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
                yield self.reformat_output(next_text, kwargs)
            return

        return token_generator()


def exl2(
    model_path: str,
    draft_model_path: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    cache_q4: bool = False,
    paged: bool = True,
    max_chunk_size: Optional[int] = None,
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
        from exllamav2 import (  # type: ignore
            ExLlamaV2,
            ExLlamaV2Cache,
            ExLlamaV2Cache_Q4,
            ExLlamaV2Config,
            ExLlamaV2Tokenizer,
        )
        from exllamav2.generator import ExLlamaV2DynamicGenerator

    except ImportError:
        raise ImportError(
            "The `exllamav2`, `transformers` and `torch` libraries needs to be installed in order to use `exllamav2` models. "
            "Please run `pip install transformers torch git+https://github.com/lapp0/exllamav2@sampler-logits-processor` "
            "Documentation: https://dottxt-ai.github.io/outlines/latest/reference/models/exllamav2/"
        )
    config = ExLlamaV2Config(model_path)
    if max_chunk_size is not None:
        config.max_input_len = max_chunk_size
        config.max_attention_size = max_chunk_size**2

    config.arch_compat_overrides()
    model = ExLlamaV2(config)
    if max_seq_len is None:
        max_seq_len = -1
    if cache_q4:
        cache = ExLlamaV2Cache_Q4(model, max_seq_len=max_seq_len, lazy=True)
    else:
        cache = ExLlamaV2Cache(model, max_seq_len=max_seq_len, lazy=True)
    model.load_autosplit(cache, progress=True)

    print("Loading tokenizer...")
    tokenizer = ExLlamaV2Tokenizer(config)
    max_batch_size = 4 if paged else 1

    draft_model = None
    draft_cache = None
    if draft_model_path is not None:
        draft_config = ExLlamaV2Config(draft_model_path)
        draft_model = ExLlamaV2(draft_config)

        if cache_q4:
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
    max_seq_len = cache.max_seq_len

    outlines_tokenizer = OutlinesExLlamaV2Tokenizer(tokenizer)
    outlines_exl2_model = ExLlamaV2Model(generator, outlines_tokenizer, max_seq_len)
    return outlines_exl2_model
