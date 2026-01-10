import sys
import logging
import pytest
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

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


@pytest.fixture(scope="session")
def hf_cache_dir(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("hf_cache")

    mp = pytest.MonkeyPatch()
    mp.setenv("HF_HOME", str(cache_dir))
    mp.setenv("HF_HUB_CACHE", str(cache_dir))
    mp.setenv("TRANSFORMERS_CACHE", str(cache_dir))

    yield str(cache_dir)

    mp.undo()

def pytest_configure():
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
