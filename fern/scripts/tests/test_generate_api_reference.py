# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
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
    cpp_utilities = (
        REPO_ROOT / "fern" / "pages" / "cpp_api" / "cpp-api-utilities.md"
    )
    cpp_thrust = (
        REPO_ROOT
        / "fern"
        / "pages"
        / "cpp_api"
        / "cpp-api-thrust-integrations.md"
    )
    python_pylibrmm = (
        REPO_ROOT
        / "fern"
        / "pages"
        / "python_api"
        / "python-api-rmm-pylibrmm.md"
    )
    python_statistics = (
        REPO_ROOT
        / "fern"
        / "pages"
        / "python_api"
        / "python-api-rmm-statistics.md"
    )

    assert cpp_index.exists()
    assert cpp_memory.exists()
    assert python_rmm.exists()
    assert "### Arena Memory Resource Class" in cpp_memory.read_text(
        encoding="utf-8"
    )
    assert "### Rmm Export" not in cpp_memory.read_text(encoding="utf-8")
    assert "Pool Memory Resource" in cpp_memory.read_text(encoding="utf-8")
    assert "reinitialize" in python_rmm.read_text(encoding="utf-8")
    assert (
        "def deallocate(self, ptr: int, nbytes: int, stream: Stream = ...) -> None:"
        in python_mr.read_text(encoding="utf-8")
    )
    utilities_text = cpp_utilities.read_text(encoding="utf-8")
    assert (
        "Determine at runtime if the CUDA driver supports the stream-ordered "
        "memory allocator functions."
    ) in utilities_text
    assert (
        "Check whether the specified `cudaMemAllocationHandleType` is supported "
        "on the present CUDA driver/runtime version."
    ) in utilities_text
    assert (
        "Determine at runtime if the CUDA driver/runtime supports the stream-ordered "
        "managed memory allocator functions."
    ) in utilities_text
    assert "### Exec Policy Class" in cpp_thrust.read_text(encoding="utf-8")
    assert "### Exec Policy Constructor" in cpp_thrust.read_text(
        encoding="utf-8"
    )
    assert (
        "### `copy_to_host` (DeviceBuffer, line 58)"
        in python_pylibrmm.read_text(encoding="utf-8")
    )
    assert (
        "using `push_statistics()` and `pop_statistics()`"
        in python_statistics.read_text(encoding="utf-8")
    )

    api_pages = sorted((REPO_ROOT / "fern" / "pages").glob("*_api/**/*.md"))
    generated_text = "\n".join(
        path.read_text(encoding="utf-8") for path in api_pages
    )
    assert ".. automodule::" not in generated_text
    assert "```{doxygengroup}" not in generated_text
    assert "* @brief" not in generated_text
    assert "* `@brief`" not in generated_text
    assert "*/" not in generated_text

    duplicate_headings = duplicate_level_three_headings(api_pages)
    assert duplicate_headings == {}


def duplicate_level_three_headings(paths: list[Path]) -> dict[str, list[str]]:
    duplicates = {}
    for path in paths:
        headings = [
            line.removeprefix("### ").split(" [`#", 1)[0]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("### ")
        ]
        counts = Counter(headings)
        repeated = sorted(
            heading for heading, count in counts.items() if count > 1
        )
        if repeated:
            duplicates[path.relative_to(REPO_ROOT).as_posix()] = repeated
    return duplicates
