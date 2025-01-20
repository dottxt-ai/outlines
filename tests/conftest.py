import sys

import pytest


def pytest_collection_modifyitems(config, items):
    if sys.platform != "linux":
        if not config.option.keyword or (
            config.option.keyword and "test_integration_vllm" in config.option.keyword
        ):
            print(
                "WARNING: test_integration_vllm tests are skipped because vLLM only supports Linux platform (including WSL)."
            )
        skip_vllm = pytest.mark.skip(reason="vLLM models can only be run on Linux.")
        for item in items:
            if "test_integration_vllm" in item.nodeid:
                item.add_marker(skip_vllm)
