import json
from typing import Any, Dict, List, Optional, Tuple

from tests.test_utils.utils import hash_dict


def normalize_for_hash(obj):
    """Normalize objects for consistent hashing.

    lms.Chat objects have unique identifiers that change between instances,
    so we convert them to a canonical dict format for hashing.
    """
    obj_str = str(obj)
    if obj_str.startswith("Chat.from_history("):
        # Get the json from the string representation
        json_part = obj_str[len("Chat.from_history("):-1]
        data = json.loads(json_part)
        return {
            "type": "lms.Chat",
            "messages": normalize_lmstudio_messages(data.get("messages", []))
        }
    elif isinstance(obj, dict):
        return {k: normalize_for_hash(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize_for_hash(item) for item in obj]
    else:
        return obj


def normalize_lmstudio_messages(messages):
    """Normalize message list for hashing."""
    result = []
    for msg in messages:
        normalized_msg = {
            "role": msg.get("role", ""),
            "content": normalize_lmstudio_content(msg.get("content", "")),
        }
        result.append(normalized_msg)
    return result


def normalize_lmstudio_content(content):
    """Normalize message content for hashing."""
    if isinstance(content, list):
        result = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    result.append({"type": "text", "text": item.get("text", "")})
                elif item.get("type") == "file":
                    result.append({"type": "file", "sizeBytes": item.get("sizeBytes", 0)})
                else:
                    result.append(item)
            else:
                result.append(str(item))
        return result
    elif isinstance(content, str):
        return content
    else:
        return str(content)


def hash_lmstudio_request(data: dict) -> str:
    """Hash a request dict, normalizing lms.Chat objects."""
    normalized = normalize_for_hash(data)
    return hash_dict(normalized)


class MockLMStudioResponse:
    """Mock for LMStudio response object"""

    def __init__(self, content: str):
        self.content = content


class MockLMStudioModel:
    """Mock for LMStudio model object returned by client.llm.model()"""

    def __init__(self, mock_responses: Dict[str, Any]):
        self._mock_responses = mock_responses

    def respond(self, messages, **kwargs):
        request_key = hash_lmstudio_request({"messages": messages, **kwargs})
        response = self._mock_responses.get(request_key)
        if not response:
            raise ValueError(f"No response found for {{'messages': {messages}, **{kwargs}}}")
        return MockLMStudioResponse(response)

    def respond_stream(self, messages, **kwargs):
        request_key = hash_lmstudio_request({"messages": messages, **kwargs})
        response = self._mock_responses.get(request_key)
        if not response:
            raise ValueError(f"No response found for {{'messages': {messages}, **{kwargs}}}")
        for chunk in response:
            yield MockLMStudioResponse(chunk)


class MockLMStudioLLM:
    """Mock for the llm attribute of Client"""

    def __init__(self, mock_responses: Dict[str, Any]):
        self._mock_responses = mock_responses

    def model(self, model_key=None):
        return MockLMStudioModel(self._mock_responses)


class MockLMStudioClient:
    """Mock for LMStudio `Client` that can be used to test the LMStudio model"""

    def __init__(self):
        self._mock_responses: Dict[str, Any] = {}
        self.llm: Optional[MockLMStudioLLM] = None

    def add_mock_responses(self, mocks: List[Tuple[dict, Any]]):
        for kwargs, response in mocks:
            request_key = hash_lmstudio_request(kwargs)
            self._mock_responses[request_key] = response
        self.llm = MockLMStudioLLM(self._mock_responses)


class MockAsyncLMStudioModel:
    """Mock for async LMStudio model object returned by client.llm.model()"""

    def __init__(self, mock_responses: Dict[str, Any]):
        self._mock_responses = mock_responses

    async def respond(self, messages, **kwargs):
        request_key = hash_lmstudio_request({"messages": messages, **kwargs})
        response = self._mock_responses.get(request_key)
        if not response:
            raise ValueError(f"No response found for {{'messages': {messages}, **{kwargs}}}")
        return MockLMStudioResponse(response)

    async def respond_stream(self, messages, **kwargs):
        """Return an async iterator (must be awaited first, then iterated)."""
        request_key = hash_lmstudio_request({"messages": messages, **kwargs})
        response = self._mock_responses.get(request_key)
        if not response:
            raise ValueError(f"No response found for {{'messages': {messages}, **{kwargs}}}")

        async def _stream():
            for chunk in response:
                yield MockLMStudioResponse(chunk)

        return _stream()


class MockAsyncLMStudioLLM:
    """Mock for the llm attribute of AsyncClient"""

    def __init__(self, mock_responses: Dict[str, Any]):
        self._mock_responses = mock_responses

    async def model(self, model_key=None):
        return MockAsyncLMStudioModel(self._mock_responses)


class MockAsyncLMStudioClient:
    """Mock for LMStudio `AsyncClient` that can be used to test the AsyncLMStudio model"""

    def __init__(self):
        self._mock_responses: Dict[str, Any] = {}
        self.llm: Optional[MockAsyncLMStudioLLM] = None
        self._context_entered = False

    def add_mock_responses(self, mocks: List[Tuple[dict, Any]]):
        for kwargs, response in mocks:
            request_key = hash_lmstudio_request(kwargs)
            self._mock_responses[request_key] = response
        self.llm = MockAsyncLMStudioLLM(self._mock_responses)

    async def __aenter__(self):
        self._context_entered = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._context_entered = False
        return False
