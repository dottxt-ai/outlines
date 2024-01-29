from pathlib import Path

GRAMMAR_PATH = Path(__file__).parent / "grammars"


def read_grammar(grammar_file_name, base_grammar_path=GRAMMAR_PATH):
    """Read grammar file from default grammar path"""
    full_path = base_grammar_path / grammar_file_name
    with open(full_path) as file:
        return file.read()


arithmetic = read_grammar("arithmetic.lark")
csv = read_grammar("csv.lark")
json = read_grammar("json.lark")
lark = read_grammar("lark.lark")
sql_select = read_grammar("sql_select.lark")
