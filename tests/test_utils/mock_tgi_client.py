from typing import Any, Dict
from unittest.mock import MagicMock

from tests.test_utils.utils import hash_dict


class MockTGIInferenceClient:
    """Mock for TGI `InferenceClient` that can be used to test the TGI model"""

    def __init__(self):
        self.text_generation = MagicMock()

        # The method that will be called by the model when it makes a request
        def _create(**kwargs):
            # Hash the arguments to create a unique key
            request_key = hash_dict(kwargs)
            response = self._mock_responses.get(request_key)
            if not response:
                raise ValueError(f"No response found for {kwargs}")
            if kwargs.get("stream", False):
                return iter(response)
            else:
                return response

        self.text_generation.side_effect = _create
        self._mock_responses: Dict[str, Any] = {}

    def add_mock_responses(self, mocks: list):
        for kwargs, response in mocks:
            request_key = hash_dict(kwargs)
            self._mock_responses[request_key] = response


class MockAsyncTGIInferenceClient:
    """Mock for TGI `InferenceClient` that can be used to test the TGI model"""

    def __init__(self):
        self.text_generation = MagicMock()

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
                return response

        self.text_generation.side_effect = _async_create
        self._mock_responses: Dict[str, Any] = {}

    def add_mock_responses(self, mocks: list):
        for kwargs, response in mocks:
            request_key = hash_dict(kwargs)
            self._mock_responses[request_key] = response

    async def _create_async_streaming_response(self, response):
        """Create an async streaming response generator"""
        for chunk in response:
            yield chunk
