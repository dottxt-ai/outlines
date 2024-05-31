import outlines

outlines.disable_cache()

from outlines.fsm.guide import RegexGuide  # noqa: E402
from outlines.fsm.json_schema import build_regex_from_schema  # noqa: E402

from .common import (  # noqa: E402
    clear_outlines_cache,
    ensure_numba_compiled,
    setup_tokenizer,
)

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


class JsonSchemaBenchmark:
    params = schemas.keys()

    def setup(self, schema_name):
        clear_outlines_cache()
        self.tokenizer = setup_tokenizer()
        self.schema = schemas[schema_name]
        ensure_numba_compiled(self.tokenizer)

    def time_json_schema_to_regex(self, schema_name):
        build_regex_from_schema(self.schema)

    def time_json_schema_to_fsm(self, schema_name):
        regex = build_regex_from_schema(self.schema)
        RegexGuide(regex, self.tokenizer)
