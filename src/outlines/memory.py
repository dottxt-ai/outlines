"""Optional memory augmentation for synchronous text generators."""

from inspect import iscoroutine, iscoroutinefunction
from typing import Any, Callable, Protocol

TextGenerator = Callable[..., str]


class MemoryBackend(Protocol):
    """Memory client interface, compatible with Dakera's synchronous client.

    Recall returns either None or an object with a ``memories`` sequence whose
    entries have a ``content`` string. No memory SDK is required by Outlines.

    """

    def recall(self, *, agent_id: str, query: str, top_k: int) -> Any:
        """Retrieve relevant memories for the given agent and query."""
        ...

    def store_memory(
        self, *, agent_id: str, content: str, metadata: dict
    ) -> Any:
        """Store a completed generation with its source metadata."""
        ...


class MemoryAugmentedGenerator:
    """Recall past outputs as context and store completed generations.

    Wrap an existing synchronous Outlines generator without changing its
    output constraints. Only string prompts and non-streaming calls are
    supported. Memory and generation errors propagate to the caller.

    """

    def __init__(
        self,
        generator: TextGenerator,
        memory: MemoryBackend,
        *,
        agent_id: str = "outlines-agent",
        top_k: int = 2,
    ):
        """
        Parameters
        ----------
        generator
            An existing synchronous generator returning generated text.
        memory
            A client implementing ``recall`` and ``store_memory``.
        agent_id
            The namespace used to recall and store memories.
        top_k
            The positive number of memories to request on each call.

        """
        operations = (generator, memory.recall, memory.store_memory)
        if any(
            iscoroutinefunction(operation)
            or iscoroutinefunction(getattr(operation, "__call__", None))
            for operation in operations
        ):
            raise TypeError("Memory augmentation requires synchronous operations")
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        self.generator = generator
        self.memory = memory
        self.agent_id = agent_id
        self.top_k = top_k

    def __call__(self, prompt: str, **inference_kwargs: Any) -> str:
        """Generate with recalled context, then store the completed output.

        Parameters
        ----------
        prompt
            The original string prompt, used as the memory query.
        **inference_kwargs
            Inference options passed unchanged to the wrapped generator.

        Returns
        -------
        str
            The original output from the wrapped generator.

        """
        if not isinstance(prompt, str):
            raise TypeError("Memory augmentation requires a string prompt")
        prior = self.memory.recall(
            agent_id=self.agent_id, query=prompt, top_k=self.top_k
        )
        augmented_prompt = prompt
        if prior is not None and prior.memories:
            examples = "\n".join(f"- {item.content}" for item in prior.memories)
            augmented_prompt = f"Prior extractions:\n{examples}\n\n{prompt}"
        result = self.generator(augmented_prompt, **inference_kwargs)
        if not isinstance(result, str):
            if iscoroutine(result):
                result.close()
            raise TypeError("Memory augmentation requires a string result")
        self.memory.store_memory(
            agent_id=self.agent_id,
            content=result,
            metadata={"source": prompt[:100]},
        )
        return result
