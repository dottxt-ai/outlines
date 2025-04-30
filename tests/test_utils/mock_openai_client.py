from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock

from tests.test_utils.utils import hash_dict


class MockChoice:
    def __init__(
        self,
        content: str,
        finish_reason: str = "stop",
        refusal: Optional[str] = None
    ):
        self.message = MagicMock()
        self.message.content = content
        self.message.refusal = refusal
        self.finish_reason = finish_reason
        self.delta = MagicMock()
        self.delta.content = content


class MockCompletionResponse:
    def __init__(self, choices: List[MockChoice]):
        self.choices = choices


class MockStreamingChunk:
    def __init__(self, content: Optional[str] = None):
        self.choices = []
        if content is not None:
            choice = MagicMock()
            delta = MagicMock()
            delta.content = content
            choice.delta = delta
            self.choices = [choice]


class MockOpenAIClient:
    """Mock for OpenAI client that can be used to test vLLM integration"""

    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock()

        # The method that will be called by the model when it makes a request
        def _create(**kwargs):
            # Hash the arguments to create a unique key
            request_key = hash_dict(kwargs)
            response = self._mock_responses.get(request_key)
            if not response:
                raise ValueError(f"No response found for {kwargs}")
            if kwargs.get("stream", False):
                return self._create_streaming_response(response)
            else:
                return self._create_standard_response(response)

        self.chat.completions.create.side_effect = _create
        self._mock_responses: Dict[str, Any] = {}

    def add_mock_responses(self, mocks: list):
        for kwargs, response in mocks:
            request_key = hash_dict(kwargs)
            self._mock_responses[request_key] = response

    def _create_standard_response(self, response):
        if isinstance(response, str):
            response = [response]
        choices = [MockChoice(content=chunk) for chunk in response]
        return MockCompletionResponse(choices=choices)

    def _create_streaming_response(self, response):
        chunks = [MockStreamingChunk(content=chunk) for chunk in response]
        return iter(chunks)


class MockAsyncOpenAIClient:
    """Mock for AsyncOpenAI client that can be used to test AsyncVLLM integration"""

    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock()

        # The method that will be called by the model when it makes a request
        async def _async_create(**kwargs):
            # Hash the arguments to create a unique key
            request_key = hash_dict(kwargs)
            response = self._mock_responses.get(request_key)
            if not response:
                raise ValueError(f"No response found for {kwargs}")
            if kwargs.get("stream", False):
                return self._create_async_streaming_response(response)
            else:
                return await self._create_async_standard_response(response)

        self.chat.completions.create.side_effect = _async_create
        self._mock_responses: Dict[str, Any] = {}

    def add_mock_responses(self, mocks: list):
        for kwargs, response in mocks:
            request_key = hash_dict(kwargs)
            self._mock_responses[request_key] = response

    async def _create_async_standard_response(self, response):
        """Create an async standard (non-streaming) response"""
        if isinstance(response, str):
            response = [response]
        choices = [MockChoice(content=chunk) for chunk in response]
        return MockCompletionResponse(choices=choices)

    async def _create_async_streaming_response(self, response):
        """Create an async streaming response generator"""
        chunks = [MockStreamingChunk(content=chunk) for chunk in response]

        for chunk in chunks:
            yield chunk
