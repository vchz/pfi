import ast
from pathlib import Path

import mkdocs_gen_files

SRC_DIR = Path("pfi")
API_ROOT = Path("api")

nav = mkdocs_gen_files.Nav()


def top_level_symbols(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(file_path))

    classes = []
    functions = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            classes.append((node.name, node.lineno))
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            functions.append((node.name, node.lineno))

    return sorted(classes), sorted(functions)


def public_exports(init_file):
    with open(init_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(init_file))

    exports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                if name != "*" and not name.startswith("_"):
                    exports.append(name)

    return exports


def write_page(md_path, object_path, edit_path):
    with mkdocs_gen_files.open(md_path, "w") as f:
        f.write(f"::: {object_path}\n")
    mkdocs_gen_files.set_edit_path(md_path.as_posix(), edit_path)


def file_defines_symbol(py_file, symbol_name):
    classes, functions = top_level_symbols(py_file)
    return any(name == symbol_name for name, _ in classes + functions)


def ordered_python_files():
    py_files = sorted(SRC_DIR.rglob("*.py"))

    root_exports = public_exports(SRC_DIR / "__init__.py")
    if not root_exports:
        return py_files

    first_symbol = root_exports[0]

    for i, py_file in enumerate(py_files):
        if py_file.stem == "__init__":
            continue
        if file_defines_symbol(py_file, first_symbol):
            return [py_file] + py_files[:i] + py_files[i + 1 :]

    return py_files


exports_by_package = {}
for init_file in SRC_DIR.rglob("__init__.py"):
    exports_by_package[init_file.parent] = public_exports(init_file)


for py_file in ordered_python_files():
    
    if py_file.stem == "__init__":
        continue

    package_dir = py_file.parent
    public_names = exports_by_package.get(package_dir, [])
    if not public_names:
        continue

    rel_pkg = package_dir.relative_to(SRC_DIR)
    module_path = SRC_DIR.name if not rel_pkg.parts else SRC_DIR.name + "." + ".".join(rel_pkg.parts)

    classes, functions = top_level_symbols(py_file)
    is_public_module = py_file.stem in public_names
    
    if is_public_module:
        for name, lineno in classes:
            rel_md = Path(*rel_pkg.parts) / py_file.stem / f"{name}.md"
            write_page(
                API_ROOT / rel_md,
                f"{module_path}.{py_file.stem}.{name}",
                f"{py_file.as_posix()}#L{lineno}",
            )
            nav[(SRC_DIR.name,) + rel_pkg.parts + (py_file.stem, name)] = rel_md.as_posix()

        for name, lineno in functions:
            rel_md = Path(*rel_pkg.parts) / py_file.stem / f"{name}.md"
            write_page(
                API_ROOT / rel_md,
                f"{module_path}.{py_file.stem}.{name}",
                f"{py_file.as_posix()}#L{lineno}",
            )
            nav[(SRC_DIR.name,) + rel_pkg.parts + (py_file.stem, name)] = rel_md.as_posix()

    else:
        for name, lineno in classes:
            if name in public_names:
                rel_md = Path(*rel_pkg.parts) / f"{name}.md"
                write_page(
                    API_ROOT / rel_md,
                    f"{module_path}.{name}",
                    f"{py_file.as_posix()}#L{lineno}",
                )
                nav[(SRC_DIR.name,) + rel_pkg.parts + (name,)] = rel_md.as_posix()

        for name, lineno in functions:
            if name in public_names:
                rel_md = Path(*rel_pkg.parts) / f"{name}.md"
                write_page(
                    API_ROOT / rel_md,
                    f"{module_path}.{name}",
                    f"{py_file.as_posix()}#L{lineno}",
                )
                nav[(SRC_DIR.name,) + rel_pkg.parts + (name,)] = rel_md.as_posix()


with mkdocs_gen_files.open(API_ROOT / "SUMMARY.md", "w") as nav_file:
    nav_file.write("# API reference\n\n")
    nav_file.write(
        "Browse the public classes and functions exposed by `pfi`.\n\n"
    )

    nav_file.writelines(nav.build_literate_nav())

