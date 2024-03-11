import pytest

import outlines

outlines.disable_cache()

from outlines.fsm.guide import RegexGuide  # noqa: E402
from outlines.fsm.json_schema import build_regex_from_schema  # noqa: E402

simple_schema = """{
        "$defs": {
            "Armor": {
                "enum": ["leather", "chainmail", "plate"],
                "title": "Armor",
                "type": "string"
            }
        },
        "properties": {
            "name": {"maxLength": 10, "title": "Name", "type": "string"},
            "age": {"title": "Age", "type": "integer"},
            "armor": {"$ref": "#/$defs/Armor"},
            "strength": {"title": "Strength", "type": "integer"}\
        },
        "required": ["name", "age", "armor", "strength"],
        "title": "Character",
        "type": "object"
    }"""


complex_schema = """{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "title": "Schema for a recording",
  "type": "object",
  "definitions": {
    "artist": {
      "type": "object",
      "properties": {
        "id": {"type": "number"},
        "name": {"type": "string"},
        "functions": {
          "type": "array",
          "items": {"type": "string"}
        }
      },
      "required": ["id", "name", "functions"]
    }
  },
  "properties": {
    "id": {"type": "number"},
    "work": {
      "type": "object",
      "properties": {
        "id": {"type": "number"},
        "name": {"type": "string"},
        "composer": {"$ref": "#/definitions/artist"}
      }
    },
    "recording_artists": {
      "type": "array",
      "items": {"$ref": "#/definitions/artist"}
    }
  },
  "required": ["id", "work", "recording_artists"]
}"""


schemas = dict(simple_schema=simple_schema, complex_schema=complex_schema)


@pytest.mark.parametrize("schema_name", schemas.keys())
def test_benchmark_json_schema_to_regex(benchmark, ensure_numba_compiled, schema_name):
    """Benchmark convert json schema to regex"""
    schema = schemas[schema_name]
    benchmark.pedantic(
        build_regex_from_schema,
        args=(schema,),
        rounds=8,
    )


@pytest.mark.parametrize("schema_name", schemas.keys())
def test_benchmark_json_schema_to_fsm(
    benchmark, tokenizer, ensure_numba_compiled, schema_name
):
    """Benchmark compile json schema as FSM"""
    schema = schemas[schema_name]
    regex = build_regex_from_schema(schema)
    benchmark.pedantic(
        RegexGuide,
        args=(regex, tokenizer),
        rounds=8,
    )
