from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from outlines import Outline


class IterableMock(MagicMock):
    def __getattr__(self, name):
        result = MagicMock()
        result.__iter__.return_value = iter([])
        return result


class OutputModel(BaseModel):
    result: int


def template(a: int) -> str:
    return f"What is 2 times {a}?"


def test_outline():
    mock_model = IterableMock()
    mock_generator = MagicMock()
    mock_generator.return_value = '{"result": 6}'
    with patch("outlines.generate.json", return_value=mock_generator):
        outline_instance = Outline(mock_model, template, OutputModel)
        assert issubclass(outline_instance.output_type, BaseModel)
        result = outline_instance(3)
    assert result.result == 6


def test_outline_with_json_schema():
    mock_model = IterableMock()
    mock_generator = MagicMock()
    mock_generator.return_value = '{"result": 6}'
    with patch("outlines.generate.json", return_value=mock_generator):
        outline_instance = Outline(
            mock_model,
            template,
            '{"type": "object", "properties": {"result": {"type": "integer"}}}',
        )
        result = outline_instance(3)
    assert result["result"] == 6


def test_invalid_output_type():
    mock_model = IterableMock()
    with pytest.raises(TypeError):
        Outline(mock_model, template, int)


def test_invalid_json_response():
    mock_model = IterableMock()
    mock_generator = MagicMock()
    mock_generator.return_value = "invalid json"
    with patch("outlines.generate.json", return_value=mock_generator):
        outline_instance = Outline(mock_model, template, OutputModel)
        with pytest.raises(ValueError, match="Unable to parse response"):
            outline_instance(3)


def test_invalid_json_schema():
    mock_model = IterableMock()
    invalid_json_schema = (
        '{"type": "object", "properties": {"result": {"type": "invalid_type"}}}'
    )
    with pytest.raises(TypeError, match="Invalid JSON Schema"):
        Outline(mock_model, template, invalid_json_schema)
