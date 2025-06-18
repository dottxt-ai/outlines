from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class SamplingParameters:
    """Sampling parameters available in Outlines."""
    sampler: str
    num_samples: int = 1
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None


class Sampler(Protocol):
    """
    Protocol for all samplers.

    As outlines now delegates the handling of the generation loop
    to the models, the samplers are only used to define the sampling
    parameters that are passed to the model.
    """

    samples: int

    def sampling_params(self) -> SamplingParameters:
        ...


class GreedySampler:
    """Greedy Sampling algorithm.

    Greedy sampling consists in choosing the token with the largest
    likelihood at every step.

    We don't allow more than one sample. We could attribute this a meaning, for
    instance the k-th sample represents the k-th most likely token. In which
    case it would be equivalent to beam search without the sequence weights.

    Attributes
    ----------
    samples
        The number of samples taken for each input sequence.

    """

    def __init__(self):
        self.samples = 1

    @property
    def sampling_params(self):
        return SamplingParameters("greedy", self.samples, None, None, 0.0)


greedy = GreedySampler


class MultinomialSampler:
    """Multinomial sampling algorithm.

    Multinomial sampling consists in randomly sampling the next token assuming
    its distribution is a Categorical distribution parametrized by the
    next-token logits.


    Attributes
    ----------
    samples
        The number of samples taken for each input sequence.

    """

    def __init__(
        self,
        samples: int = 1,
        *,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
    ):
        self.samples = samples
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

    @property
    def sampling_params(self):
        return SamplingParameters(
            "multinomial",
            self.samples,
            self.top_p,
            self.top_k,
            self.temperature,
        )


multinomial = MultinomialSampler


class BeamSearchSampler:
    """Beam Search sampling algorithm.

    Attributes
    ----------
    samples
        The number of samples taken for each input sequence. Equivalent to the
        number of beams.
    """

    def __init__(self, beams: int = 1):
        self.samples = beams

    @property
    def sampling_params(self):
        return SamplingParameters("beam_search", self.samples, None, None, 1.0)


beam_search = BeamSearchSampler
