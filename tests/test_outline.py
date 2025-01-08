from unittest.mock import Mock, patch

from pydantic import BaseModel

from outlines import Outline


class OutputModel(BaseModel):
    result: int


def template(a: int) -> str:
    return f"What is 2 times {a}?"


def test_outline():
    mock_model = Mock()
    mock_generator = Mock()
    mock_generator.return_value = '{"result": 6}'

    with patch("outlines.generate.json", return_value=mock_generator):
        outline_instance = Outline(mock_model, template, OutputModel)
        result = outline_instance(3)

    assert result.result == 6


def test_outline_with_json_schema():
    mock_model = Mock()
    mock_generator = Mock()
    mock_generator.return_value = '{"result": 6}'

    with patch("outlines.generate.json", return_value=mock_generator):
        outline_instance = Outline(
            mock_model,
            template,
            '{"type": "object", "properties": {"result": {"type": "integer"}}}',
        )
        result = outline_instance(3)

    assert result["result"] == 6
