# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "fern" / "scripts" / "generate_api_reference.py"


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_api_reference", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_restored_sphinx_sources_keep_api_semantics():
    cpp_source = (
        REPO_ROOT / "fern" / "sphinx" / "source" / "cpp" / "data_containers.md"
    ).read_text(encoding="utf-8")
    python_source = (
        REPO_ROOT / "fern" / "sphinx" / "source" / "python" / "mr.md"
    ).read_text(encoding="utf-8")
    conf = (REPO_ROOT / "fern" / "sphinx" / "source" / "conf.py").read_text(
        encoding="utf-8"
    )

    assert "{doxygengroup} data_containers" in cpp_source
    assert ".. automodule:: rmm.mr" in python_source
    assert "sphinx_markdown_builder" in conf
    assert "../../../cpp/doxygen/xml" in conf


def test_copy_generated_markdown_pages_preserves_rendered_content(tmp_path):
    generator = load_generator()
    markdown_dir = tmp_path / "sphinx" / "build" / "markdown"
    output_dir = tmp_path / "fern" / "pages" / "api_reference"
    (markdown_dir / "cpp").mkdir(parents=True)
    (markdown_dir / "python").mkdir(parents=True)
    (markdown_dir / "cpp" / "data_containers.md").write_text(
        "# Data Containers\n\n"
        "RAII construct for device memory allocation.\n\n"
        "This class allocates untyped and uninitialized device memory.\n",
        encoding="utf-8",
    )
    (markdown_dir / "python" / "mr.md").write_text(
        "# rmm.mr (Memory Resources)\n\n"
        "Python memory resource docstrings are rendered here.\n",
        encoding="utf-8",
    )

    copied = generator.copy_generated_markdown_pages(markdown_dir, output_dir)

    cpp_output = (output_dir / "cpp" / "data_containers.md").read_text(
        encoding="utf-8"
    )
    python_output = (output_dir / "python" / "mr.md").read_text(
        encoding="utf-8"
    )
    assert copied == 2
    assert "Generated from the Sphinx API extraction build." in cpp_output
    assert "This class allocates untyped and uninitialized device memory." in cpp_output
    assert "Python memory resource docstrings are rendered here." in python_output
    assert "docs.rapids.ai/api/rmm" not in cpp_output + python_output


def test_docs_yml_includes_native_api_navigation():
    docs_yml = (REPO_ROOT / "fern" / "docs.yml").read_text(encoding="utf-8")

    assert 'section: "API Reference"' in docs_yml
    assert "./pages/api_reference/cpp/data_containers.md" in docs_yml
    assert "./pages/api_reference/python/mr.md" in docs_yml
