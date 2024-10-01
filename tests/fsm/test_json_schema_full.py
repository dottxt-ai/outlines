# LOCAL IMPORT HACK
import importlib
import json
import re

import pytest
import requests
import requests_cache
import yaml
from referencing.exceptions import Unresolvable

from outlines.fsm.json_schema import build_regex_from_schema

dump_yaml_normalized = importlib.import_module("test_json_schema").dump_yaml_normalized


requests_cache.install_cache("test_request_cache", expire_after=3600)


def get_json_schema_tests_from_repo(
    repo="json-schema-org/JSON-Schema-Test-Suite", configs_dir="tests/draft2020-12"
):
    api_url = f"https://api.github.com/repos/{repo}/contents/{configs_dir}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    contents = response.json()

    results = []
    for item in contents:
        if item["type"] == "file" and item["name"].endswith(".json"):
            file_url = item["download_url"]
            file_response = requests.get(file_url)
            file_response.raise_for_status()
            json_data = file_response.json()

            for entry in json_data:
                for test in entry["tests"]:
                    results.append(
                        {
                            "file": item["name"],
                            "schema": json.dumps(entry["schema"]),
                            "data": json.dumps(test["data"]),
                            "is_valid": test["valid"],
                        }
                    )

    return results


@pytest.mark.skip("Utility for improving compliance with json schema spec")
@pytest.mark.parametrize("sample", get_json_schema_tests_from_repo())
def test_json_schema_to_json_compliance(sample):
    """
    Assert that we either correctly handle a schema, or skip if NotImplementedError
    """
    try:
        pattern = build_regex_from_schema(
            sample["schema"], strict_json_schema_subset=False
        )
    except NotImplementedError as e:
        pytest.skip(f"{e}")
    except Unresolvable:
        pytest.xfail()

    if sample["is_valid"]:
        assert (
            re.fullmatch(pattern, sample["data"]) is not None
        ), "Failed to match valid schema"
    else:
        assert (
            re.fullmatch(pattern, sample["data"]) is None
        ), "Incorrectly matched invalid schema"


@pytest.mark.parametrize("sample", get_json_schema_tests_from_repo())
def test_json_schema_to_yaml_compliance(sample):
    """
    Skip tests checking whether it can be built, that is covered by
    `test_json_schema_compliance`.

    Here we are purely testing whether yaml is valid IFF json schema is valid.
    """
    # skip if the test fails to construct the pattern
    try:
        json_pattern = build_regex_from_schema(
            sample["schema"], strict_json_schema_subset=False
        )
    except Exception:
        pytest.skip()

    # skip invalid with json
    json_valid = re.fullmatch(json_pattern, sample["data"]) is not None
    if sample["is_valid"] != json_valid:
        pytest.skip()

    # valide yaml generation for samples for all samples where json is valid
    try:
        yaml_pattern = build_regex_from_schema(
            sample["schema"], mode="yaml", strict_json_schema_subset=False
        )
    except NotImplementedError as e:
        pytest.skip(f"{e}")
    except Unresolvable:
        pytest.xfail()

    yaml_sample = dump_yaml_normalized(json.loads(sample["data"]))

    # xfail complex mappings
    if any(line.startswith("? ") for line in yaml_sample.split("\n")):
        pytest.xfail()

    if sample["is_valid"]:
        assert (
            re.fullmatch(yaml_pattern, yaml_sample) is not None
        ), "Failed to match valid schema"
    else:
        assert (
            re.fullmatch(yaml_pattern, yaml_sample) is None
        ), "Incorrectly matched invalid schema"


@pytest.mark.skip()
@pytest.mark.parametrize("sample", get_json_schema_tests_from_repo())
def test_yaml_dumper_consistency(sample):
    """valide output yaml is equivalent to input json"""
    sample_from_json = json.loads(sample["data"])
    sample_from_yaml = yaml.safe_load(dump_yaml_normalized(sample_from_json))
    assert sample_from_yaml == sample_from_json
    assert json.dumps(sample_from_yaml) == sample["data"]
