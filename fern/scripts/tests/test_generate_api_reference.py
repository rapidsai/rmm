# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR = REPO_ROOT / "fern" / "scripts" / "generate_api_reference.py"


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_api_reference", GENERATOR
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generator_writes_cpp_and_python_api_pages():
    generator = load_generator()

    assert generator.main() == 0

    cpp_index = REPO_ROOT / "fern" / "pages" / "cpp_api" / "index.md"
    cpp_memory = (
        REPO_ROOT
        / "fern"
        / "pages"
        / "cpp_api"
        / "cpp-api-memory-resources.md"
    )
    python_rmm = (
        REPO_ROOT / "fern" / "pages" / "python_api" / "python-api-rmm.md"
    )
    python_mr = (
        REPO_ROOT / "fern" / "pages" / "python_api" / "python-api-rmm-mr.md"
    )

    assert cpp_index.exists()
    assert cpp_memory.exists()
    assert python_rmm.exists()
    assert "### Arena Memory Resource" in cpp_memory.read_text(
        encoding="utf-8"
    )
    assert "### Rmm Export" not in cpp_memory.read_text(encoding="utf-8")
    assert "Pool Memory Resource" in cpp_memory.read_text(encoding="utf-8")
    assert "reinitialize" in python_rmm.read_text(encoding="utf-8")
    assert (
        "def deallocate(self, ptr: int, nbytes: int, stream: Stream = ...) -> None:"
        in python_mr.read_text(encoding="utf-8")
    )

    generated_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [cpp_index, cpp_memory, python_rmm, python_mr]
    )
    assert ".. automodule::" not in generated_text
    assert "```{doxygengroup}" not in generated_text
