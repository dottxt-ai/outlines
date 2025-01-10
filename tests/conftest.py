import sys

import pytest


def pytest_collection_modifyitems(config, items):
    if sys.platform != "linux":
        skip_vllm = pytest.mark.skip(reason="vLLM models can only be run on Linux.")
        for item in items:
            if "test_integration_vllm" in item.nodeid:
                item.add_marker(skip_vllm)
                print(
                    f"WARNING: {item.nodeid} is skipped because vLLM only supports Linux platform (including WSL)."
                )
