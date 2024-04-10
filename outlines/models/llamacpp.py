import dataclasses
from typing import TYPE_CHECKING, Iterator, List, Optional, TypedDict, Union

from typing_extensions import Unpack

from outlines.generate.api import GenerationParameters, SamplingParameters

if TYPE_CHECKING:
    from llama_cpp import Llama, LogitsProcessorList


class LlamaCppParams(TypedDict, total=False):
    suffix: Optional[str]
    temperature: float
    top_p: float
    min_p: float
    typical_p: float
    seed: int
    max_tokens: int
    logits_processor: "LogitsProcessorList"
    stop: Optional[Union[str, List[str]]]
    frequence_penalty: float
    presence_penalty: float
    repeat_penalty: float
    top_k: int
    tfs_z: float
    mirostat_mode: int
    mirostat_tau: float
    mirostat_eta: float
    stream: bool


class LlamaCpp:
    """Represents a model provided by the `llama-cpp-python` library.

    We wrap models from model providing libraries in order to give all of
    them the same interface in Outlines and allow users to easily switch
    between providers. This class wraps the `llama_cpp.Llama` class from the
    `llama-cpp-python` library.

    """

    def __init__(self, model: "Llama"):
        self.model = model

    def prepare_generation_parameters(
        self,
        generation_parameters: GenerationParameters,
        sampling_parameters: SamplingParameters,
        structure_logits_processor,
        **llama_cpp_params: Unpack[LlamaCppParams],
    ):
        """Prepare the generation parameters.

        `llama-cpp-python` uses different default values

        """
        from llama_cpp import LogitsProcessorList

        max_tokens, stop_at, seed = dataclasses.astuple(generation_parameters)

        # We update `llama_cpp_params` with the values the user passed to the
        # generator.
        if "stop" not in llama_cpp_params:
            llama_cpp_params["stop"] = stop_at
        if "seed" not in llama_cpp_params:
            llama_cpp_params["seed"] = seed

        # Somehow `llama-cpp-python` generates `max_tokens + 1`  tokens
        if "max_tokens" not in llama_cpp_params:
            if max_tokens is not None:
                llama_cpp_params["max_tokens"] = max_tokens - 1
        else:
            llama_cpp_params["max_tokens"] = llama_cpp_params["max_tokens"] - 1

        sampler, num_samples, top_p, top_k, temperature = dataclasses.astuple(
            sampling_parameters
        )

        # We update the `llama_cpp_params` with the sampling values that
        # were specified by the user via the `Sampler` class, unless they
        # are also specified in `llama_cpp_params`. We also disable other
        # sampling methods that are enabled by default and reset the temperature
        # value.
        #
        # See https://github.com/ggerganov/llama.cpp/blob/e11a8999b5690f810c2c99c14347f0834e68c524/common/sampling.h#L22
        # for the default values in `llama.cpp` and indications to disable the sampling modes.
        # Mirostat sampling, tail-free sampling and all penalties are disabled by default.
        #
        # See https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__call__
        # for default values in `llama-cpp-python`
        if sampler == "beam_search":
            raise NotImplementedError(
                "The `llama_cpp_python` library does not support Beam Search."
            )
        if num_samples != 1:
            raise NotImplementedError(
                "The `llama_cpp_python` library does not allow to take several samples."
            )
        if "top_p" not in llama_cpp_params:
            if top_p is not None:
                llama_cpp_params["top_p"] = top_p
            else:
                llama_cpp_params["top_p"] = 1.0

        if "min_p" not in llama_cpp_params:
            llama_cpp_params["min_p"] = 0.0

        if "top_k" not in llama_cpp_params:
            if top_k is not None:
                llama_cpp_params["top_k"] = top_k
            else:
                llama_cpp_params["top_k"] = -1

        if "temperature" not in llama_cpp_params:
            if temperature is not None:
                llama_cpp_params["temperature"] = temperature
            else:
                llama_cpp_params["temperature"] = 1.0

        if "repeat_penalty" not in llama_cpp_params:
            llama_cpp_params["repeat_penalty"] = 1.0

        # The choice to stream or not should happen via the high-level API
        llama_cpp_params["stream"] = False

        if structure_logits_processor is not None:
            if "logits_processor" in llama_cpp_params:
                llama_cpp_params["logits_processor"].append(structure_logits_processor)
            else:
                llama_cpp_params["logits_processor"] = LogitsProcessorList(
                    [structure_logits_processor]
                )

        return llama_cpp_params

    def generate(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        structure_logits_processor,
        sampling_parameters: SamplingParameters,
        **llama_cpp_params: Unpack[LlamaCppParams],
    ) -> str:
        """Generate text using `llama-cpp-python`.

        Arguments
        ---------
        prompts
            A prompt or list of prompts.
        generation_parameters
            An instance of `GenerationParameters` that contains the prompt,
            the maximum number of tokens, stop sequences and seed. All the
            arguments to `SequenceGeneratorAdapter`'s `__cal__` method.
        logits_processor
            The logits processor to use when generating text.
        sampling_parameters
            An instance of `SamplingParameters`, a dataclass that contains
            the name of the sampler to use and related parameters as available
            in Outlines.
        llama_cpp_params
            Keyword arguments that can be passed to
            `llama_cpp_python.Llama.__call__`.  The values in `llama_cpp_params`
            supersede the values of the parameters in `generation_parameters` and
            `sampling_parameters`.  See the `llama_cpp_python` documentation for
            a list of possible values: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__call__

        Returns
        -------
        The generated text.

        """
        if not isinstance(prompts, str):
            raise NotImplementedError(
                "The `llama-cpp-python` library does not support batch inference."
            )

        llama_cpp_params = self.prepare_generation_parameters(
            generation_parameters,
            sampling_parameters,
            structure_logits_processor,
            **llama_cpp_params,
        )
        completion = self.model(prompts, **llama_cpp_params)
        result = completion["choices"][0]["text"]

        self.model.reset()

        return result

    def stream(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        structure_logits_processor,
        sampling_parameters: SamplingParameters,
        **llama_cpp_params: Unpack[LlamaCppParams],
    ) -> Iterator[str]:
        """Stream text using `llama-cpp-python`.

        Arguments
        ---------
        prompts
            A prompt or list of prompts.
        generation_parameters
            An instance of `GenerationParameters` that contains the prompt,
            the maximum number of tokens, stop sequences and seed. All the
            arguments to `SequenceGeneratorAdapter`'s `__cal__` method.
        logits_processor
            The logits processor to use when generating text.
        sampling_parameters
            An instance of `SamplingParameters`, a dataclass that contains
            the name of the sampler to use and related parameters as available
            in Outlines.
        llama_cpp_params
            Keyword arguments that can be passed to
            `llama_cpp_python.Llama.__call__`.  The values in `llama_cpp_params`
            supersede the values of the parameters in `generation_parameters` and
            `sampling_parameters`.  See the `llama_cpp_python` documentation for
            a list of possible values: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__call__

        Returns
        -------
        A generator that return strings.

        """

        if not isinstance(prompts, str):
            raise NotImplementedError(
                "The `llama-cpp-python` library does not support batch inference."
            )

        llama_cpp_params = self.prepare_generation_parameters(
            generation_parameters,
            sampling_parameters,
            structure_logits_processor,
            **llama_cpp_params,
        )
        llama_cpp_params["stream"] = True
        generator = self.model(prompts, **llama_cpp_params)

        def token_generator() -> Iterator[str]:
            while True:
                try:
                    result = next(generator)
                    yield result["choices"][0]["text"]
                except StopIteration:
                    self.model.reset()
                    return

        return token_generator()

    def load_lora(self, adapter_path: str):
        if self.model._model.apply_lora_from_file(
            adapter_path,
            1.0,
        ):
            raise RuntimeError(f"Failed to apply LoRA from lora path: {adapter_path}")


def llamacpp(
    repo_id: str, filename: Optional[str] = None, **llamacpp_model_params
) -> LlamaCpp:
    """Load a model from the `llama-cpp-python` library.

    We use the `Llama.from_pretrained` classmethod that downloads models
    directly from the HuggingFace hub, instead of asking users to specify
    a path to the downloaded model. One can still load a local model
    by initializing `llama_cpp.Llama` directly.

    Arguments
    ---------
    repo_id
        The name of the model repository.
    filename:
        A filename of glob pattern to match the model file in the repo.
    llama_cpp_model_params
        Llama-specific model parameters. See the `llama-cpp-python` documentation
        for the full list: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__

    """
    from llama_cpp import Llama

    # Default to using the model's full context length
    if "n_ctx" not in llamacpp_model_params:
        llamacpp_model_params["n_ctx"] = 0

    if "verbose" not in llamacpp_model_params:
        llamacpp_model_params["verbose"] = False

    model = Llama.from_pretrained(repo_id, filename, **llamacpp_model_params)

    return LlamaCpp(model)
