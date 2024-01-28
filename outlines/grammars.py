from pathlib import Path
import re


GRAMMAR_PATH = Path(__file__).parent / "grammars"


def read_grammar(grammar_file_name, base_grammar_path=GRAMMAR_PATH):
    """Read grammar file from default grammar path"""
    full_path = base_grammar_path / grammar_file_name
    with open(full_path) as file:
        return file.read()


def codeblocks(grammar_text):
    pattern = r'^(?:\?start|start)(.*)$'
    result, n = re.subn(
        pattern,
        lambda m: (
            '?codeblock_inner_start' + m.group(1)
            if m.group(0).startswith('?start')
            else 'codeblock_inner_start' + m.group(1)
        ),
        grammar_text,
        flags=re.MULTILINE
    )
    assert n == 1, "Grammar not formatted correctly for codeblock wrapping"
    result = r"""
    ?start: "```\n" codeblock_inner_start "\n```"
    """ + result
    return result



arithmetic = read_grammar("arithmetic.lark")
csv = read_grammar("csv.lark")
json = read_grammar("json.lark")
lark = read_grammar("lark.lark")
sql_select = read_grammar("sql_select.lark")
