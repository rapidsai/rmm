#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Generate Fern API reference pages from RMM source files."""

from __future__ import annotations

import ast
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

REPO_DIR = Path(__file__).resolve().parents[2]
FERN_PAGES = REPO_DIR / "fern" / "pages"
CPP_INCLUDE = REPO_DIR / "cpp" / "include" / "rmm"
PYTHON_RMM = REPO_DIR / "python" / "rmm" / "rmm"
PYTHON_LIBRMM = REPO_DIR / "python" / "librmm" / "librmm"


@dataclass(frozen=True)
class CppSection:
    title: str
    slug: str
    headers: tuple[str, ...]


@dataclass(frozen=True)
class PythonSection:
    title: str
    slug: str
    sources: tuple[Path, ...]


@dataclass
class CppEntry:
    name: str
    kind: str
    signature: str
    summary: str
    source: Path
    line: int


@dataclass
class PythonEntry:
    name: str
    kind: str
    signature: str
    doc: str
    source: Path
    line: int
    parent: str | None = None


CPP_SECTIONS = [
    CppSection(
        "Memory Resources",
        "memory-resources",
        (
            "mr/*memory_resource.hpp",
            "mr/per_device_resource.hpp",
            "mr/polymorphic_allocator.hpp",
            "mr/system_memory_resource.hpp",
        ),
    ),
    CppSection(
        "Memory Resource Adaptors",
        "memory-resource-adaptors",
        (
            "mr/*adaptor.hpp",
            "mr/callback_memory_resource.hpp",
            "mr/failure_callback_t.hpp",
            "mr/statistics_resource_adaptor.hpp",
            "mr/tracking_resource_adaptor.hpp",
        ),
    ),
    CppSection(
        "Data Containers",
        "data-containers",
        (
            "device_buffer.hpp",
            "device_scalar.hpp",
            "device_uvector.hpp",
            "device_vector.hpp",
            "resource_ref.hpp",
        ),
    ),
    CppSection(
        "Thrust Integrations", "thrust-integrations", ("exec_policy.hpp",)
    ),
    CppSection(
        "CUDA Device Management",
        "cuda-device-management",
        ("cuda_device.hpp", "prefetch.hpp", "process_is_exiting.hpp"),
    ),
    CppSection(
        "CUDA Streams",
        "cuda-streams",
        ("cuda_stream.hpp", "cuda_stream_pool.hpp", "cuda_stream_view.hpp"),
    ),
    CppSection("Errors", "errors", ("error.hpp",)),
    CppSection(
        "Utilities",
        "utilities",
        ("aligned.hpp", "logger.hpp", "detail/runtime_capabilities.hpp"),
    ),
]

PYTHON_SECTIONS = [
    PythonSection(
        "rmm", "rmm", (PYTHON_RMM / "rmm.py", PYTHON_RMM / "__init__.py")
    ),
    PythonSection(
        "rmm.mr",
        "rmm-mr",
        (
            PYTHON_RMM / "mr" / "__init__.py",
            PYTHON_RMM / "mr" / "experimental.py",
            PYTHON_RMM
            / "pylibrmm"
            / "memory_resource"
            / "_memory_resource.pyi",
            PYTHON_RMM / "pylibrmm" / "memory_resource" / "experimental.pyi",
        ),
    ),
    PythonSection(
        "rmm.allocators",
        "rmm-allocators",
        (
            PYTHON_RMM / "allocators" / "cupy.py",
            PYTHON_RMM / "allocators" / "numba.py",
            PYTHON_RMM / "allocators" / "torch.py",
        ),
    ),
    PythonSection(
        "rmm.statistics", "rmm-statistics", (PYTHON_RMM / "statistics.py",)
    ),
    PythonSection(
        "rmm.pylibrmm",
        "rmm-pylibrmm",
        tuple(sorted((PYTHON_RMM / "pylibrmm").rglob("*.pyi"))),
    ),
    PythonSection(
        "rmm.librmm",
        "rmm-librmm",
        (PYTHON_LIBRMM / "__init__.py", PYTHON_LIBRMM / "load.py"),
    ),
]

COMMENT_RE = re.compile(r"/\*\*.*?\*/", re.DOTALL)


def main() -> int:
    reset_generated_pages()
    generate_cpp_pages()
    generate_python_pages()
    return 0


def reset_generated_pages() -> None:
    for directory in [FERN_PAGES / "cpp_api", FERN_PAGES / "python_api"]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def generate_cpp_pages() -> None:
    out_dir = FERN_PAGES / "cpp_api"
    index_lines = [
        *frontmatter("api-reference/cpp-api-index"),
        "# C++ API",
        "",
        "These pages are generated from documented RMM public headers.",
        "",
    ]

    for section in CPP_SECTIONS:
        route = f"cpp-api-{section.slug}"
        index_lines.append(f"- [{section.title}](/api-reference/{route})")
        entries_by_header = collect_cpp_entries(section)
        lines = [
            *frontmatter(f"api-reference/{route}"),
            f"# {section.title}",
            "",
            "Generated from RMM C++ headers.",
            "",
        ]
        if not entries_by_header:
            lines.extend(["No documented public declarations were found.", ""])
        heading_counts = Counter(
            cpp_heading_base(entry)
            for entries in entries_by_header.values()
            for entry in entries
        )
        heading_seen: dict[str, int] = {}
        for header, entries in entries_by_header.items():
            lines.extend([f"## `{header}`", ""])
            if not entries:
                lines.extend(["No documented declarations found.", ""])
                continue
            for entry in entries:
                heading = unique_cpp_heading(
                    entry, heading_counts, heading_seen
                )
                lines.extend(render_cpp_entry(entry, heading))
        write_page(out_dir / f"{route}.md", lines)

    write_page(out_dir / "index.md", index_lines)
    write_page(out_dir / "cpp-api-namespaces.md", cpp_namespaces_page())


def collect_cpp_entries(section: CppSection) -> dict[str, list[CppEntry]]:
    entries_by_header: dict[str, list[CppEntry]] = {}
    headers = unique_paths(
        header
        for pattern in section.headers
        for header in sorted(CPP_INCLUDE.glob(pattern))
        if header.is_file()
    )
    for header in headers:
        entries_by_header[relative(header)] = parse_cpp_header(header)
    return entries_by_header


def parse_cpp_header(path: Path) -> list[CppEntry]:
    text = read_text(path)
    entries: list[CppEntry] = []
    for match in COMMENT_RE.finditer(text):
        comment = clean_doxygen_comment(match.group(0))
        if (
            not comment
            or "@file" in comment
            or is_doxygen_group_marker(comment)
        ):
            continue
        declaration, line = read_cpp_declaration(text, match.end())
        if not declaration:
            continue
        name = cpp_declaration_name(declaration)
        if not name:
            continue
        entries.append(
            CppEntry(
                name=name,
                kind=cpp_declaration_kind(declaration),
                signature=normalize_cpp_signature(declaration),
                summary=brief_from_comment(comment),
                source=path,
                line=line,
            )
        )
    return entries


def read_cpp_declaration(text: str, start: int) -> tuple[str, int]:
    suffix = text[start:]
    line = text[:start].count("\n") + 1
    lines: list[str] = []
    brace_depth = 0
    in_block_comment = False
    for raw_line in suffix.splitlines():
        stripped = raw_line.strip()
        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
            line += 1
            continue
        if (
            not stripped
            or stripped.startswith("//")
            or stripped.startswith("/*")
        ):
            if stripped.startswith("/*") and "*/" not in stripped:
                in_block_comment = True
            line += 1
            continue
        lines.append(stripped)
        brace_depth += stripped.count("{") - stripped.count("}")
        if (
            stripped.endswith(";")
            or stripped.endswith("{}")
            or ("{" in stripped and "}" in stripped and brace_depth == 0)
            or (stripped.endswith("}") and brace_depth == 0)
            or (brace_depth > 0 and "{" in stripped)
        ):
            break
        if len(lines) >= 8:
            break
    declaration = " ".join(lines).strip()
    return declaration, line


def cpp_declaration_name(signature: str) -> str:
    signature = signature.strip()
    patterns = [
        r"^(?:class|struct|enum(?:\s+class)?)\s+(?:[A-Z_][A-Z0-9_]*\s+)*([A-Za-z_]\w*)",
        r"^using\s+([A-Za-z_]\w*)\s*=",
        r"^(?:inline\s+)?(?:constexpr\s+)?(?:auto|[\w:<>~*&\s]+)\s+([A-Za-z_]\w*)\s*\(",
        r"^([A-Za-z_]\w*)\s*\(",
    ]
    for pattern in patterns:
        if match := re.search(pattern, signature):
            return humanize_symbol(match.group(1))
    return ""


def cpp_declaration_kind(signature: str) -> str:
    signature = signature.strip()
    if re.match(r"^class\b", signature):
        return "Class"
    if re.match(r"^struct\b", signature):
        return "Struct"
    if re.match(r"^enum(?:\s+class)?\b", signature):
        return "Enum"
    if re.match(r"^using\b", signature):
        return "Type Alias"
    if re.match(
        r"^(?:explicit\s+)?(?:constexpr\s+)?(?:inline\s+)?~[A-Za-z_]\w*\s*\(",
        signature,
    ):
        return "Destructor"
    if re.match(
        r"^(?:explicit\s+)?(?:constexpr\s+)?(?:inline\s+)?[A-Za-z_]\w*\s*\(",
        signature,
    ):
        return "Constructor"
    return ""


def cpp_heading_base(entry: CppEntry) -> str:
    if entry.kind in {"Class", "Struct", "Enum", "Constructor", "Destructor"}:
        return f"{entry.name} {entry.kind}"
    if entry.kind == "Type Alias":
        return f"{entry.name} Type Alias"
    return entry.name


def unique_cpp_heading(
    entry: CppEntry,
    heading_counts: Counter[str],
    heading_seen: dict[str, int],
) -> str:
    base = cpp_heading_base(entry)
    if heading_counts[base] == 1:
        return base
    heading_seen[base] = heading_seen.get(base, 0) + 1
    return f"{base} ({entry.source.name}:{entry.line})"


def render_cpp_entry(entry: CppEntry, heading: str) -> list[str]:
    lines = [f"### {heading}", ""]
    if entry.summary:
        lines.extend([entry.summary, ""])
    lines.extend(["```cpp", entry.signature, "```", ""])
    lines.extend([f"_Source: `{relative(entry.source)}:{entry.line}`_", ""])
    return lines


def cpp_namespaces_page() -> list[str]:
    return [
        *frontmatter("api-reference/cpp-api-namespaces"),
        "# Namespaces",
        "",
        "## `rmm`",
        "",
        "RAPIDS Memory Manager - the top-level namespace for RMM functionality.",
        "",
        "## `rmm::mr`",
        "",
        "Memory resource classes and adaptors for CUDA memory allocation strategies.",
        "",
    ]


def generate_python_pages() -> None:
    out_dir = FERN_PAGES / "python_api"
    index_lines = [
        *frontmatter("api-reference/python-api-index"),
        "# Python API",
        "",
        "These pages are generated from RMM Python source files and type stubs.",
        "",
    ]

    for section in PYTHON_SECTIONS:
        route = f"python-api-{section.slug}"
        index_lines.append(f"- [{section.title}](/api-reference/{route})")
        lines = [
            *frontmatter(f"api-reference/{route}"),
            f"# {section.title}",
            "",
            "Generated from RMM Python sources.",
            "",
        ]
        entries_by_source = []
        for source in section.sources:
            if not source.exists():
                continue
            entries = parse_python_source(source)
            if not entries:
                continue
            entries_by_source.append((source, entries))
        heading_counts = Counter(
            python_heading_base(entry)
            for _, entries in entries_by_source
            for entry in entries
        )
        heading_seen: dict[str, int] = {}
        for source, entries in entries_by_source:
            lines.extend([f"## `{relative(source)}`", ""])
            for entry in entries:
                heading = unique_python_heading(
                    entry, heading_counts, heading_seen
                )
                lines.extend(render_python_entry(entry, heading))
        write_page(out_dir / f"{route}.md", lines)

    write_page(out_dir / "index.md", index_lines)


def parse_python_source(path: Path) -> list[PythonEntry]:
    if path.suffix == ".pyi":
        return parse_pyi_source(path)
    return parse_py_source(path)


def parse_py_source(path: Path) -> list[PythonEntry]:
    tree = ast.parse(read_text(path), filename=str(path))
    entries: list[PythonEntry] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            entries.append(
                PythonEntry(
                    name=node.name,
                    kind="class",
                    signature=f"class {node.name}",
                    doc=ast.get_docstring(node) or "",
                    source=path,
                    line=node.lineno,
                    parent=None,
                )
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            entries.append(
                PythonEntry(
                    name=node.name,
                    kind="function",
                    signature=python_function_signature(node),
                    doc=ast.get_docstring(node) or "",
                    source=path,
                    line=node.lineno,
                    parent=None,
                )
            )
    return entries


def parse_pyi_source(path: Path) -> list[PythonEntry]:
    entries: list[PythonEntry] = []
    lines = read_text(path).splitlines()
    index = 0
    current_class: str | None = None
    while index < len(lines):
        line_number = index + 1
        raw_line = lines[index]
        stripped = raw_line.strip()
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if (
            indent == 0
            and stripped
            and not stripped.startswith(("class ", "@"))
        ):
            current_class = None
        if stripped.startswith("class "):
            name = (
                stripped.removeprefix("class ")
                .split("(", 1)[0]
                .split(":", 1)[0]
            )
            current_class = name
            entries.append(
                PythonEntry(
                    name,
                    "class",
                    stripped.rstrip(":"),
                    "",
                    path,
                    line_number,
                    parent=None,
                )
            )
        elif stripped.startswith("def ") and not stripped.startswith("def _"):
            name = stripped.removeprefix("def ").split("(", 1)[0]
            declaration = [stripped]
            while index + 1 < len(lines) and not pyi_def_is_complete(
                declaration[-1]
            ):
                index += 1
                declaration.append(lines[index].strip())
            signature = normalize_pyi_signature(" ".join(declaration))
            entries.append(
                PythonEntry(
                    name,
                    "function",
                    signature,
                    "",
                    path,
                    line_number,
                    parent=current_class if indent > 0 else None,
                )
            )
        index += 1
    return entries


def pyi_def_is_complete(line: str) -> bool:
    return line.endswith(":") or line.endswith(": ...")


def normalize_pyi_signature(signature: str) -> str:
    signature = normalize_cpp_signature(signature.removesuffix("...").rstrip())
    return signature.replace("( ", "(").replace(" )", ")")


def python_function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = [arg.arg for arg in node.args.args]
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    args.extend(arg.arg for arg in node.args.kwonlyargs)
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    return f"{prefix} {node.name}({', '.join(args)})"


def python_heading_base(entry: PythonEntry) -> str:
    if entry.kind == "function" and entry.parent:
        return f"`{entry.name}` ({entry.parent})"
    return f"`{entry.name}`"


def unique_python_heading(
    entry: PythonEntry,
    heading_counts: Counter[str],
    heading_seen: dict[str, int],
) -> str:
    base = python_heading_base(entry)
    if heading_counts[base] == 1:
        return base
    heading_seen[base] = heading_seen.get(base, 0) + 1
    context = entry.parent or entry.source.stem
    return f"`{entry.name}` ({context}, line {entry.line})"


def render_python_entry(entry: PythonEntry, heading: str) -> list[str]:
    lines = [
        f"### {heading}",
        "",
        "```python",
        entry.signature,
        "```",
        "",
    ]
    if entry.doc:
        lines.extend(render_docstring(entry.doc))
        lines.append("")
    lines.extend([f"_Source: `{relative(entry.source)}:{entry.line}`_", ""])
    return lines


def clean_doxygen_comment(raw: str) -> str:
    text = raw.strip()
    text = text.removeprefix("/**").removesuffix("*/")
    cleaned = []
    for line in text.splitlines():
        cleaned.append(re.sub(r"^\s*\* ?", "", line).rstrip())
    return "\n".join(cleaned).strip()


def is_doxygen_group_marker(comment: str) -> bool:
    return any(
        marker in comment
        for marker in ("@addtogroup", "\\addtogroup", "@{", "\\{", "@}", "\\}")
    )


def brief_from_comment(comment: str) -> str:
    lines = [line.strip() for line in comment.splitlines()]
    for index, line in enumerate(lines):
        if match := re.match(r"[@\\]brief\s*(.*)", line):
            return convert_doxygen_text(
                " ".join(brief_continuation(lines, index, match.group(1)))
            )
    first_paragraph = []
    for line in lines:
        if not line:
            if first_paragraph:
                break
            continue
        if is_doxygen_command(line):
            if first_paragraph:
                break
            continue
        first_paragraph.append(line)
    if first_paragraph:
        return convert_doxygen_text(" ".join(first_paragraph))
    return ""


def brief_continuation(
    lines: list[str],
    start_index: int,
    first_line: str,
) -> list[str]:
    continuation = [first_line] if first_line else []
    for line in lines[start_index + 1 :]:
        if not line:
            if continuation:
                break
            continue
        if is_doxygen_command(line):
            break
        continuation.append(line)
    return continuation


def is_doxygen_command(line: str) -> bool:
    return bool(re.match(r"^[@\\][A-Za-z]", line))


def convert_doxygen_text(value: str) -> str:
    value = re.sub(r"[@\\]briefreturn\{([^}]+)\}", r"Returns \1.", value)
    value = re.sub(r"[@\\](?:p|c)\s+([A-Za-z_]\w*)", r"`\1`", value)
    value = re.sub(r"[@\\]\w+\b", " ", value)
    value = value.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", value).strip()


def normalize_cpp_signature(signature: str) -> str:
    return re.sub(r"\s+", " ", signature).strip()


def render_docstring(doc: str) -> list[str]:
    lines = []
    for line in doc.splitlines():
        stripped = line.rstrip()
        if stripped.startswith("Parameters") or set(stripped) == {"-"}:
            continue
        lines.append(stripped)
    return trim_blank(lines)


def frontmatter(slug: str) -> list[str]:
    return ["---", f"slug: {slug}", "---", ""]


def write_page(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(trim_blank(lines)).rstrip() + "\n", encoding="utf-8"
    )
    print(f"Wrote {relative(path)}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def relative(path: Path) -> str:
    return path.relative_to(REPO_DIR).as_posix()


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    return list(dict.fromkeys(paths))


def humanize_symbol(value: str) -> str:
    words = value.replace("_", " ").split()
    special = {
        "cuda": "CUDA",
        "rmm": "RMM",
        "mr": "MR",
    }
    return " ".join(special.get(word, word.capitalize()) for word in words)


def trim_blank(lines: list[str]) -> list[str]:
    lines = list(lines)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
