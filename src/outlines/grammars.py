"""A few common Lark grammars."""

from pathlib import Path

GRAMMAR_PATH = Path(__file__).parent / "grammars"


def read_grammar(
    grammar_file_name: str,
    base_grammar_path: Path = GRAMMAR_PATH,
) -> str:
    """Read grammar file from default grammar path.

    Parameters
    ----------
    grammar_file_name
        The name of the grammar file to read.
    base_grammar_path
        The path to the directory containing the grammar file.

    Returns
    -------
    str
        The contents of the grammar file.

    """
    full_path = base_grammar_path / grammar_file_name
    with open(full_path) as file:
        return file.read()


arithmetic = read_grammar("arithmetic.lark")
json = read_grammar("json.lark")
