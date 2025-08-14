import re

import llama_cpp
import outlines
import pytest
import transformers
from xgrammar import GrammarCompiler, TokenizerInfo

from outlines.backends.xgrammar import XGrammarBackend, XGrammarLogitsProcessor
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
        )
    )

def model_mlxlm():
    return outlines.from_mlxlm(
        *mlx_lm.load("mlx-community/SmolLM-135M-Instruct-4bit")
    )

@pytest.fixture
def tokenizer_info():
    tokenizer = model_transformers().hf_tokenizer
    tokenizer_info = TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=len(tokenizer.get_vocab())
    )
    return tokenizer_info

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
def cfg():
    return """
root ::= answer
answer ::= "yes" | "no"
"""


def test_xgr_processor_torch(regex):
    model = model_transformers()
    tokenizer = model.tokenizer
    hf_tokenizer = model.hf_tokenizer
    tokenizer_info = TokenizerInfo.from_huggingface(
        hf_tokenizer,
        vocab_size=len(hf_tokenizer.get_vocab())
    )
    grammar_compiler = GrammarCompiler(tokenizer_info)
    compiled_grammar = grammar_compiler.compile_regex(regex)
    processor = XGrammarLogitsProcessor(compiled_grammar, "torch")
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


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_xgr_processor_mlx(tokenizer_info):
    model = model_mlxlm()
    tokenizer = model.mlx_tokenizer
    tokenizer_info = TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=len(tokenizer.get_vocab())
    )
    grammar_compiler = GrammarCompiler(tokenizer_info)
    compiled_grammar = grammar_compiler.compile_regex(regex)
    processor = XGrammarLogitsProcessor(compiled_grammar, "mlx")
    for _ in range(2):
        input_ids = simulate_model_calling_processor(
            processor,
            "mlx",
            len(tokenizer.get_vocab()),
            tokenizer.eos_token_id,
            2
        )
        assert re.match(regex, tokenizer.decode(input_ids[0]))
        assert re.match(regex, tokenizer.decode(input_ids[1]))


models = [(model_transformers(), "torch")]
if HAS_MLX:
    models.append((model_mlxlm(), "mlx"))

@pytest.mark.parametrize("model, tensor_library_name", models)
def test_xgrammar_backend(model, tensor_library_name, json_schema, regex, cfg):
    # initialization
    backend = XGrammarBackend(model)
    assert isinstance(backend.grammar_compiler, GrammarCompiler)

    # json schema
    processor = backend.get_json_schema_logits_processor(json_schema)
    assert isinstance(processor, XGrammarLogitsProcessor)
    generator = outlines.Generator(model, backend="xgrammar", processor=processor)
    response = generator("Hello, how are you?")
    assert response[0] == "{"
    assert "name" in response

    # regex
    processor = backend.get_regex_logits_processor(regex)
    assert isinstance(processor, XGrammarLogitsProcessor)
    generator = outlines.Generator(model, backend="xgrammar", processor=processor)
    response = generator("Hello, how are you?")
    assert len(response) == 3
    assert int(response)

    # cfg
    processor = backend.get_cfg_logits_processor(cfg)
    assert isinstance(processor, XGrammarLogitsProcessor)
    generator = outlines.Generator(model, backend="xgrammar", processor=processor)
    response = generator("Hello, how are you?")
    assert response == "yes" or response == "no"

    # batch + multiple generations
    processor = backend.get_json_schema_logits_processor(json_schema)
    generator = outlines.Generator(model, backend="xgrammar", processor=processor)
    for _ in range(2):
        if tensor_library_name == "torch":
            response = generator.batch(["Create a character", "Hello, how are you?"], max_new_tokens=200)
            assert len(response) == 2
            for r in response:
                assert r[0] == "{"
                assert "name" in r
        else:
            response = generator("Create a character", max_tokens=20)
            assert response[0] == "{"
            assert "name" in response


def test_xgrammar_backend_invalid_model():
    with pytest.raises(
        ValueError,
        match="The xgrammar backend only supports Transformers and MLXLM models",
    ):
        XGrammarBackend(model_llamacpp())
