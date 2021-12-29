"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("funfact").glob("**/*.py")):
    module_path = path.relative_to(".").with_suffix("")
    if '_ast' in str(module_path):
        continue
    doc_path = path.relative_to(".").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = list(module_path.parts)
    parts[-1] = f"{parts[-1]}.py"
    nav[parts] = doc_path

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(module_path.parts)
        print("::: " + ident, file=fd)
        print("::: " + ident)

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    print('NAV\n', nav.build_literate_nav(), sep='')
    nav_file.writelines(nav.build_literate_nav())
