import pytest


# Run @pytest.mark.benchmark_cfg if --benchmark-cfg is passed
# Otherwise, run everything else
def pytest_addoption(parser):
    parser.addoption(
        "--benchmark-cfg", action="store_true", help="Only run CFG benchmarks."
    )


@pytest.fixture
def benchmark_cfg(request):
    """A fixture to check if the benchmark tests should be run."""
    return request.config.getoption("--benchmark-cfg")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "benchmark_cfg: mark test as a benchmark to run only when --benchmark-cfg is provided.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--benchmark-cfg"):
        # Keep only benchmark tests
        keep = [item for item in items if "benchmark_cfg" in item.keywords]
    else:
        # Keep only non-benchmark tests
        keep = [item for item in items if "benchmark_cfg" not in item.keywords]

    items[:] = keep  # Replace the items list with the filtered list
