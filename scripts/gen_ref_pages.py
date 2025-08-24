"""Generate the API reference pages and navigation automatically.

This script is based on the `gen_ref_pages.py` script in the
[mkdocstrings](https://mkdocstrings.github.io/recipes/#automatic-code-reference-pages) project.

To exclude a file or module from being included in the generated API reference,
add a part of its path to the `EXCLUDED_FILES` list.
"""

from pathlib import Path

import mkdocs_gen_files

CODEBASE_DIR_NAME = "outlines"
OUTPUT_DIR_NAME = "api_reference"
EXCLUDED_FILES = ["_version"]


nav = mkdocs_gen_files.Nav()
root = Path(__file__).parent.parent
src = root / CODEBASE_DIR_NAME

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path(OUTPUT_DIR_NAME, doc_path)

    parts = tuple(module_path.parts)

    if any(part in EXCLUDED_FILES for part in parts):
        continue

    if parts[-1] == "__init__":
        if len(parts) == 1:
            doc_path = Path("index.md")
            full_doc_path = Path(OUTPUT_DIR_NAME, doc_path)
            parts = (CODEBASE_DIR_NAME,)
        else:
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        if len(parts) == 1 and parts[0] == CODEBASE_DIR_NAME:
            # For root module, just use the package name
            fd.write(f"::: {CODEBASE_DIR_NAME}")
        else:
            fd.write(f"::: {CODEBASE_DIR_NAME}.{ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open(f"{OUTPUT_DIR_NAME}/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
