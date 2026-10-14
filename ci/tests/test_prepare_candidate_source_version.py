# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for preparing the disposable candidate source version."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "ci/prepare_candidate_source_version.py"
VERSION_READER = REPOSITORY_ROOT / "cmake/rapids_config.cmake"


def _source_version() -> str:
    return (REPOSITORY_ROOT / "VERSION").read_text(encoding="utf-8").strip()


def _next_candidate_version() -> str:
    major, minor, patch = _source_version().split(".")
    return f"{major}.{int(minor) + 2:02}.{patch}"


def _run_helper(
    directory: Path, *, candidate: str, versions: dict[str, object] | None
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["RAPIDS_RELEASE_CANDIDATE"] = candidate
    environment.pop("RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS", None)
    if versions is not None:
        environment["RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS"] = json.dumps(
            versions
        )
    return subprocess.run(
        ["python", str(SCRIPT)],
        cwd=directory,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_frozen_unit_version_drives_real_cmake_reader(tmp_path: Path) -> None:
    source_version = _source_version()
    candidate_version = _next_candidate_version()
    (tmp_path / "VERSION").write_text(f"{source_version}\n", encoding="utf-8")
    result = _run_helper(
        tmp_path,
        candidate="true",
        versions={
            "package_versions": {"conda": source_version},
            "unit_package_versions": {"conda:rmm": candidate_version},
        },
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "VERSION").read_text(encoding="utf-8") == (
        f"{candidate_version}\n"
    )

    cmake_dir = tmp_path / "cmake"
    cmake_dir.mkdir()
    shutil.copy2(VERSION_READER, cmake_dir / "rapids_config.cmake")
    (cmake_dir / "RAPIDS.cmake").write_text(
        "# Keep this fixture local; the version reader is the production file.\n",
        encoding="utf-8",
    )
    (tmp_path / "RAPIDS_BRANCH").write_text(
        f"branch-{candidate_version[:5]}\n", encoding="utf-8"
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "\n".join(
            (
                "cmake_minimum_required(VERSION 3.25)",
                "project(rmm_version_probe NONE)",
                'include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/rapids_config.cmake")',
                'file(WRITE "${CMAKE_BINARY_DIR}/rapids-version.txt" "${RAPIDS_VERSION}\\n")',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    configured = subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "build")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert configured.returncode == 0, configured.stderr
    assert (tmp_path / "build/rapids-version.txt").read_text(
        encoding="utf-8"
    ) == f"{candidate_version}\n"


@pytest.mark.parametrize(
    "versions",
    [
        None,
        {},
        {"package_versions": {"wheel": _next_candidate_version()}},
        {"unit_package_versions": {"conda:rmm": "invalid"}},
    ],
)
def test_candidate_requires_valid_frozen_rmm_version(
    tmp_path: Path, versions: dict[str, object] | None
) -> None:
    original = f"{_source_version()}\n"
    (tmp_path / "VERSION").write_text(original, encoding="utf-8")

    result = _run_helper(tmp_path, candidate="true", versions=versions)

    assert result.returncode == 1
    assert (
        "Unable to prepare frozen RMM candidate source version"
        in result.stderr
    )
    assert (tmp_path / "VERSION").read_text(encoding="utf-8") == original


def test_non_candidate_build_preserves_checked_in_version(
    tmp_path: Path,
) -> None:
    original = f"{_source_version()}\n"
    (tmp_path / "VERSION").write_text(original, encoding="utf-8")

    result = _run_helper(
        tmp_path,
        candidate="false",
        versions={
            "unit_package_versions": {"conda:rmm": _next_candidate_version()}
        },
    )

    assert result.returncode == 0
    assert (tmp_path / "VERSION").read_text(encoding="utf-8") == original
