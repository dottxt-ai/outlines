from outlines.v0_legacy.samplers import (
    BeamSearchSampler,
    GreedySampler,
    MultinomialSampler,
    beam_search,
    greedy,
    multinomial,
)


def test_aliases():
    assert greedy == GreedySampler
    assert multinomial == MultinomialSampler
    assert beam_search == BeamSearchSampler


def test_greedy():
    sampler = GreedySampler()
    params = sampler.sampling_params
    assert params.sampler == "greedy"
    assert params.num_samples == 1
    assert params.top_p is None
    assert params.top_k is None
    assert params.temperature == 0.0


def test_multinomial():
    sampler = MultinomialSampler()
    params = sampler.sampling_params
    assert params.sampler == "multinomial"
    assert params.num_samples == 1
    assert params.top_p is None
    assert params.top_k is None
    assert params.temperature is None

    sampler = multinomial(samples=5, top_k=10, top_p=0.9, temperature=0.8)
    params = sampler.sampling_params
    assert params.sampler == "multinomial"
    assert params.num_samples == 5
    assert params.top_p == 0.9
    assert params.top_k == 10
    assert params.temperature == 0.8


def test_beam_search():
    sampler = BeamSearchSampler()
    params = sampler.sampling_params
    assert params.sampler == "beam_search"
    assert params.num_samples == 1
    assert params.top_p is None
    assert params.top_k is None
    assert params.temperature == 1.0

    sampler = beam_search(beams=3)
    params = sampler.sampling_params
    assert params.sampler == "beam_search"
    assert params.num_samples == 3
    assert params.top_p is None
    assert params.top_k is None
    assert params.temperature == 1.0
