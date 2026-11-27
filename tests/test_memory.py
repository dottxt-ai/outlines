from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from outlines import Generator, MemoryAugmentedGenerator
from outlines.generator import AsyncBlackBoxGenerator, BlackBoxGenerator
from outlines.models import VLLM
from tests.test_utils.mock_openai_client import MockOpenAIClient


class Memory:
    def __init__(self, prior=None):
        self.prior = prior
        self.queries = []
        self.stored = []

    def recall(self, **kwargs):
        self.queries.append(kwargs)
        return self.prior

    def store_memory(self, **kwargs):
        self.stored.append(kwargs)


class Model:
    def __init__(self):
        self.inputs = []

    def generate(self, prompt, output_type, **kwargs):
        self.inputs.append((prompt, output_type, kwargs))
        return '{"category": "billing"}'


def test_recall_and_store():
    memory = Memory(SimpleNamespace(memories=[
        SimpleNamespace(content='{"category": "billing"}'),
        SimpleNamespace(content='{"category": "technical"}'),
    ]))
    model = Model()
    schema = {"type": "object"}
    generator = MemoryAugmentedGenerator(
        BlackBoxGenerator(model, schema), memory, agent_id="tickets", top_k=3
    )

    result = generator("Classify this invoice", max_tokens=50)

    assert result == '{"category": "billing"}'
    assert model.inputs == [(
        'Prior extractions:\n- {"category": "billing"}\n'
        '- {"category": "technical"}\n\nClassify this invoice',
        schema,
        {"max_tokens": 50},
    )]
    assert memory.queries == [{
        "agent_id": "tickets", "query": "Classify this invoice", "top_k": 3,
    }]
    assert memory.stored == [{
        "agent_id": "tickets", "content": result,
        "metadata": {"source": "Classify this invoice"},
    }]


@pytest.mark.parametrize("prior", [None, SimpleNamespace(memories=[])])
def test_empty_recall_preserves_prompt(prior):
    memory = Memory(prior)
    model = Model()
    generator = MemoryAugmentedGenerator(BlackBoxGenerator(model, str), memory)
    generator("original")
    assert model.inputs == [("original", str, {})]
    assert memory.queries == [{
        "agent_id": "outlines-agent", "query": "original", "top_k": 2,
    }]


def test_source_metadata_is_bounded():
    memory = Memory()
    generator = MemoryAugmentedGenerator(BlackBoxGenerator(Model(), str), memory)
    generator("x" * 200)
    assert memory.stored[0]["metadata"] == {"source": "x" * 100}


def test_failed_generation_is_not_stored():
    def fail(prompt):
        raise ValueError("generation failed")

    memory = Memory()
    generator = MemoryAugmentedGenerator(fail, memory)
    with pytest.raises(ValueError, match="generation failed"):
        generator("input")
    assert memory.stored == []


@pytest.mark.parametrize("operation", ["recall", "store_memory"])
def test_memory_errors_are_propagated(operation):
    class UnavailableMemory(Memory):
        def recall(self, **kwargs):
            if operation == "recall":
                raise RuntimeError("memory unavailable")
            return super().recall(**kwargs)

        def store_memory(self, **kwargs):
            raise RuntimeError("memory unavailable")

    generator = MemoryAugmentedGenerator(
        BlackBoxGenerator(Model(), str), UnavailableMemory()
    )
    with pytest.raises(RuntimeError, match="memory unavailable"):
        generator("input")


@pytest.mark.parametrize("prompt", [None, 1, ["input"]])
def test_non_text_prompt_is_rejected(prompt):
    memory = Memory()
    generator = MemoryAugmentedGenerator(BlackBoxGenerator(Model(), str), memory)
    with pytest.raises(TypeError, match="string"):
        generator(prompt)
    assert memory.queries == []


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, "2"])
def test_invalid_recall_count(top_k):
    with pytest.raises(ValueError, match="positive integer"):
        MemoryAugmentedGenerator(BlackBoxGenerator(Model(), str), Memory(), top_k=top_k)


@pytest.mark.parametrize("agent_id", ["", None, 1])
def test_invalid_agent_id(agent_id):
    with pytest.raises(ValueError, match="non-empty string"):
        MemoryAugmentedGenerator(BlackBoxGenerator(Model(), str), Memory(), agent_id=agent_id)


def test_async_generator_is_rejected():
    with pytest.raises(TypeError, match="synchronous"):
        MemoryAugmentedGenerator(AsyncBlackBoxGenerator(Model(), str), Memory())


def test_async_function_is_rejected():
    async def generate(prompt):
        return "output"

    with pytest.raises(TypeError, match="synchronous"):
        MemoryAugmentedGenerator(generate, Memory())


@pytest.mark.parametrize("operation", ["recall", "store_memory"])
def test_async_memory_is_rejected(operation):
    async def unavailable(**kwargs):
        return None

    memory = Memory()
    setattr(memory, operation, unavailable)
    with pytest.raises(TypeError, match="synchronous"):
        MemoryAugmentedGenerator(BlackBoxGenerator(Model(), str), memory)


@pytest.mark.parametrize("result", [["one", "two"], {"category": "billing"}, None])
def test_non_text_result_is_not_stored(result):
    memory = Memory()
    generator = MemoryAugmentedGenerator(lambda prompt: result, memory)
    with pytest.raises(TypeError, match="string result"):
        generator("input")
    assert memory.stored == []


def test_coroutine_result_is_closed_and_not_stored():
    async def generate():
        return "output"

    result = generate()
    memory = Memory()
    generator = MemoryAugmentedGenerator(lambda prompt: result, memory)
    with pytest.raises(TypeError, match="string result"):
        generator("input")
    assert result.cr_frame is None
    assert memory.stored == []


def test_repeated_calls_recall_previous_results():
    class RecallingMemory(Memory):
        def recall(self, **kwargs):
            return SimpleNamespace(memories=[
                SimpleNamespace(content=item["content"])
                for item in self.stored
                if item["agent_id"] == kwargs["agent_id"]
            ])

    memory = RecallingMemory()
    model = Model()
    generator = MemoryAugmentedGenerator(BlackBoxGenerator(model, str), memory)
    first = generator("first")
    generator("second")
    assert model.inputs[1][0] == f"Prior extractions:\n- {first}\n\nsecond"


def test_generator_factory_preserves_json_constraints():
    class Category(BaseModel):
        category: str

    memory = Memory(SimpleNamespace(memories=[
        SimpleNamespace(content='{"category": "technical"}'),
    ]))
    client = MockOpenAIClient()
    result = '{"category": "billing"}'
    client.add_mock_responses([(
        {
            "messages": [{
                "role": "user",
                "content": 'Prior extractions:\n- {"category": "technical"}\n\nClassify invoice',
            }],
            "model": "test-model",
            "max_tokens": 50,
            "extra_body": {"structured_outputs": {"json": Category.model_json_schema()}},
        },
        result,
    )])
    generator = MemoryAugmentedGenerator(
        Generator(VLLM(client, "test-model"), Category), memory
    )
    generated = generator("Classify invoice", max_tokens=50)
    assert Category.model_validate_json(generated).category == "billing"
    assert memory.stored[0]["content"] == result
