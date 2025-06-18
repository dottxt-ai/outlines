import warnings

import pytest

try:
    import exllamav2  # type: ignore
    HAS_EXV2 = True
except ImportError:
    HAS_EXV2 = False
    warnings.warn(
        "Skipping exllamav2 legacy tests because exllamav2 is not available."
    )

from outlines import models, samplers, generate
from outlines.v0_legacy.generate.api import GeneratorV0Adapter

pytestmark = pytest.mark.skipif(
    not HAS_EXV2,
    reason="Exllamav2 is not available."
)


@pytest.fixture(scope="session")
def model():
    with pytest.warns(
        DeprecationWarning,
        match="The `exllamav2` function is deprecated",
    ):
        model = models.exl2(
            model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
            max_seq_len=1024,
            cache_q4=False,
        )
    return model


def test_exllamav2_legacy_init(model):
    assert isinstance(model, models.ExllamaV2)
    assert hasattr(model, "generator")
    assert hasattr(model, "tokenizer")
    assert model.tokenizer.convert_token_to_string(1) == 1
    assert hasattr(model, "max_seq_len")
    assert isinstance(model.max_seq_len, int)


def test_exllamav2_legacy_call(model):
    # minimal
    generator = generate.text(model)
    assert isinstance(generator, GeneratorV0Adapter)

    result = generator("Respond with a single word.")
    assert isinstance(result, str)

    # with args
    generator = generate.text(
        model,
        samplers.multinomial(samples=2, temperature=10, top_p=0.5, top_k=2)
    )
    assert isinstance(generator, GeneratorV0Adapter)

    result = generator(["Write a short story.", "Write a poem."], 2, "e", 2)
    assert isinstance(result, list)
    for sequence in result:
        assert isinstance(sequence, list)
        for s in sequence:
            assert isinstance(s, str)
            assert len(s) < 20

    # stream
    generator = generate.text(model)
    assert isinstance(generator, GeneratorV0Adapter)

    result = generator.stream("Respond with a single word.", 2)
    assert isinstance(result, GeneratorV0Adapter)
    for r in result:
        assert isinstance(r, str)
