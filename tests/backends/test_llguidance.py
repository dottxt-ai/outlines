import re

import llama_cpp
import llguidance
import pytest
import transformers
from llguidance import LLTokenizer

import outlines
from outlines.backends.llguidance import (
    LLGuidanceBackend,
    LLGuidanceLogitsProcessor
)
from tests.backends.test_backends_utils import simulate_model_calling_processor

try:
    import mlx_lm
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def model_transformers():
    return outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained("erwanf/gpt2-mini"),
        transformers.AutoTokenizer.from_pretrained("erwanf/gpt2-mini"),
    )

def model_llamacpp():
    return outlines.from_llamacpp(
        llama_cpp.Llama.from_pretrained(
            repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
            filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
            chat_format="qwen",
        )
    )

def model_mlxlm():
    return outlines.from_mlxlm(
        *mlx_lm.load("mlx-community/SmolLM-135M-Instruct-4bit")
    )

@pytest.fixture
def json_schema():
    return (
        '{"type": "object", "properties": {"name": {"type": "string"}, '
        + '"age": {"type": "integer"}}, "required": ["name", "age"], '
        + '"additionalProperties": false}'
    )

@pytest.fixture
def regex():
    return r"[0-9]{3}"

@pytest.fixture
def cfg_lark():
    return """
?start: sum

?sum: product
| sum "+" product   -> add
| sum "-" product   -> sub

?product: atom
| product "*" atom  -> mul
| product "/" atom  -> div

?atom: NUMBER           -> number
| "-" atom         -> neg
| "(" sum ")"

%import common.NUMBER
%import common.WS_INLINE

%ignore WS_INLINE
"""

@pytest.fixture
def cfg_ebnf():
    return """
root ::= answer
answer ::= "yes" | "no"
"""


def test_llguidance_processor_torch(regex):
    model = model_transformers()
    tokenizer = model.tokenizer
    hf_tokenizer = model.hf_tokenizer
    llg_tokenizer = LLGuidanceBackend(model).llg_tokenizer
    grammar_spec = llguidance.grammar_from("regex", regex)
    processor = LLGuidanceLogitsProcessor(grammar_spec, llg_tokenizer, "torch")
    for _ in range(2):
        input_ids = simulate_model_calling_processor(
            processor,
            "torch",
            len(tokenizer.get_vocab()),
            tokenizer.eos_token_id,
            2
        )
        assert re.match(regex, hf_tokenizer.decode(input_ids[0]))
        assert re.match(regex, hf_tokenizer.decode(input_ids[1]))


def test_llguidance_processor_numpy(regex):
    model = model_llamacpp()
    tokenizer = model.tokenizer
    llg_tokenizer = LLGuidanceBackend(model).llg_tokenizer
    grammar_spec = llguidance.grammar_from("regex", regex)
    processor = LLGuidanceLogitsProcessor(grammar_spec, llg_tokenizer, "numpy")
    for _ in range(2):
        input_ids = simulate_model_calling_processor(
            processor,
            "numpy",
            len(tokenizer.vocabulary),
            tokenizer.eos_token_id,
            2
        )
        assert re.match(regex, tokenizer.decode(input_ids[0])[0])
        assert re.match(regex, tokenizer.decode(input_ids[1])[0])



@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_llguidance_processor_mlx(regex):
    model = model_mlxlm()
    tokenizer = model.mlx_tokenizer
    llg_tokenizer = LLGuidanceBackend(model).llg_tokenizer
    grammar_spec = llguidance.grammar_from("regex", regex)
    processor = LLGuidanceLogitsProcessor(grammar_spec, llg_tokenizer, "mlx")
    for _ in range(2):
        input_ids = simulate_model_calling_processor(
            processor,
            "mlx",
            len(tokenizer.vocabulary),
            tokenizer.eos_token_id,
            2
        )
        assert re.match(regex, tokenizer.decode(input_ids[0]))
        assert re.match(regex, tokenizer.decode(input_ids[1]))


models = [
    (model_transformers(), "torch"),
    (model_llamacpp(), "numpy"),
]
if HAS_MLX:
    models.append((model_mlxlm(), "mlx"))

@pytest.mark.parametrize("model, tensor_library_name", models)
def test_llguidance_backend(model, tensor_library_name, json_schema, regex, cfg_lark, cfg_ebnf):
    # initialization
    backend = LLGuidanceBackend(model)
    assert isinstance(backend.llg_tokenizer, LLTokenizer)
    assert backend.tensor_library_name == tensor_library_name

    # json schema
    processor = backend.get_json_schema_logits_processor(json_schema)
    assert isinstance(processor, LLGuidanceLogitsProcessor)
    generator = outlines.Generator(model, backend="llguidance", processor=processor)
    response = generator("Hello, how are you?")
    assert response[0] == "{"

    # regex
    processor = backend.get_regex_logits_processor(regex)
    assert isinstance(processor, LLGuidanceLogitsProcessor)
    generator = outlines.Generator(model, backend="llguidance", processor=processor)
    response = generator("Hello, how are you?")
    assert len(response) == 3
    assert int(response)

    # cfg lark
    processor = backend.get_cfg_logits_processor(cfg_lark)
    assert isinstance(processor, LLGuidanceLogitsProcessor)
    generator = outlines.Generator(model, backend="llguidance", processor=processor)
    response = generator("Hello, how are you?")
    assert (
        "+" in response
        or "-" in response
        or "*" in response
        or "/" in response
        or float(response.strip())
    )

    # cfg ebnf
    processor = backend.get_cfg_logits_processor(cfg_ebnf)
    assert isinstance(processor, LLGuidanceLogitsProcessor)
    generator = outlines.Generator(model, backend="llguidance", processor=processor)
    response = generator("Hello, how are you?")
    assert response == "yes" or response == "no"

    # batch + multiple generations
    processor = backend.get_json_schema_logits_processor(json_schema)
    generator = outlines.Generator(model, backend="llguidance", processor=processor)
    for _ in range(2):
        if tensor_library_name == "torch":
            response = generator.batch(["Create a character", "Hello, how are you?"], max_new_tokens=200)
            assert len(response) == 2
            for r in response:
                assert r[0] == "{"
        else:
            response = generator("Create a character", max_tokens=20)
            assert response[0] == "{"


def test_json_schema_logits_processor_rejects_whitespace_pattern():
    """The llguidance backend does not support whitespace_pattern and raises."""
    backend = object.__new__(LLGuidanceBackend)
    schema = '{"type": "object"}'
    with pytest.raises(NotImplementedError, match="whitespace_pattern"):
        backend.get_json_schema_logits_processor(schema, whitespace_pattern=" ")


def test_cfg_logits_processor_rejects_literal_matching_special_token():
    """A grammar literal exactly matching a special token (e.g. the `<think>`
    tag on reasoning-model tokenizers) can make llguidance's parser fail with
    an opaque `ParserTooComplex` error at generation time, since the model
    may emit the literal as a single atomic special token that the parser
    can't match against a literal-string terminal. The backend should catch
    this at grammar-compile time instead. See #1771.
    """
    model = model_transformers()
    backend = LLGuidanceBackend(model)
    grammar = '?start: "<|endoftext|>" "yes"'
    with pytest.raises(ValueError, match="<\\|endoftext\\|>"):
        backend.get_cfg_logits_processor(grammar)


def test_cfg_logits_processor_allows_literal_not_matching_special_token(cfg_ebnf):
    """A grammar with no literal matching a special token compiles as usual."""
    model = model_transformers()
    backend = LLGuidanceBackend(model)
    processor = backend.get_cfg_logits_processor(cfg_ebnf)
    assert isinstance(processor, LLGuidanceLogitsProcessor)


def test_cfg_logits_processor_rejects_literal_matching_added_non_special_token():
    """Reasoning-model tokenizers commonly register tokens like `<think>` as
    added-but-not-special (`special=False`), e.g. Qwen3-4B-Thinking, so that
    `skip_special_tokens=True` decoding doesn't strip them. Such a token is
    absent from `all_special_tokens` but is still emitted atomically by the
    tokenizer, so it conflicts with a grammar literal exactly like a true
    special token does. This is the exact scenario reported in #1771.
    """
    model = model_transformers()
    model.hf_tokenizer.add_tokens(["<think>"], special_tokens=False)
    assert "<think>" not in model.hf_tokenizer.all_special_tokens
    backend = LLGuidanceBackend(model)
    grammar = '?start: "<think>" "yes"'
    with pytest.raises(ValueError, match="<think>"):
        backend.get_cfg_logits_processor(grammar)


def test_cfg_logits_processor_rejects_single_quoted_literal_matching_special_token():
    """Lark accepts single- and double-quoted string literals interchangeably,
    so the check must scan both quote styles, not just double quotes.
    """
    model = model_transformers()
    backend = LLGuidanceBackend(model)
    grammar = "?start: '<|endoftext|>' \"yes\""
    with pytest.raises(ValueError, match="<\\|endoftext\\|>"):
        backend.get_cfg_logits_processor(grammar)


def test_cfg_logits_processor_rejects_escaped_literal_matching_special_token():
    """A literal's source text can differ from the text it denotes (e.g. a
    \\uXXXX escape), so the check must decode escapes before comparing
    against token text rather than comparing raw source characters.
    """
    model = model_transformers()
    backend = LLGuidanceBackend(model)
    grammar = '?start: "\\u003c|endoftext|\\u003e" "yes"'
    with pytest.raises(ValueError, match="<\\|endoftext\\|>"):
        backend.get_cfg_logits_processor(grammar)


def test_cfg_logits_processor_ignores_literal_inside_comment(cfg_ebnf):
    """A special token's text appearing inside a `//` comment is not grammar
    text and must not trigger a false-positive rejection.
    """
    model = model_transformers()
    backend = LLGuidanceBackend(model)
    grammar = f'// example: "<|endoftext|>"\n{cfg_ebnf}'
    processor = backend.get_cfg_logits_processor(grammar)
    assert isinstance(processor, LLGuidanceLogitsProcessor)


def test_cfg_logits_processor_allows_literal_containing_special_token_as_url():
    """A `//` inside a string literal (e.g. a URL scheme) is part of the
    literal's text, not the start of a comment, and must not be truncated.
    """
    model = model_transformers()
    backend = LLGuidanceBackend(model)
    grammar = '?start: "http://example.com" "yes"'
    processor = backend.get_cfg_logits_processor(grammar)
    assert isinstance(processor, LLGuidanceLogitsProcessor)
