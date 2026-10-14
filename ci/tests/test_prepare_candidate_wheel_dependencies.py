# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the candidate-only frozen RMM wheel requirement rewrite."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPOSITORY_ROOT / "ci/prepare_candidate_wheel_dependencies.py"


def _write_pyproject(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """[build-system]
requires = ["rapids-build-backend>=0.4.0,<0.5.0"]

[project]
dependencies = ["librmm==26.10.*,>=0.0.0a0"]

[tool.rapids-build-backend]
requires = ["librmm==26.10.*,>=0.0.0a0"]
""",
        encoding="utf-8",
    )


def _run_helper(root: Path, versions: dict[str, object] | None) -> subprocess.CompletedProcess[str]:
    target = root / "python/rmm/pyproject.toml"
    _write_pyproject(target)
    helper = root / "ci/prepare_candidate_wheel_dependencies.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HELPER, helper)
    environment = os.environ.copy()
    environment.pop("RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS", None)
    if versions is not None:
        environment["RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS"] = json.dumps(versions)
    return subprocess.run(
        [sys.executable, str(helper)],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_candidate_rewrites_runtime_and_build_backend_requirements(tmp_path: Path) -> None:
    result = _run_helper(
        tmp_path,
        {
            "package_versions": {"wheel": "26.12.0"},
            "unit_package_versions": {"wheel:rmm": "26.12.0"},
        },
    )

    assert result.returncode == 0, result.stderr
    updated = (tmp_path / "python/rmm/pyproject.toml").read_text(encoding="utf-8")
    assert updated.count("librmm==26.12.0") == 2
    assert "librmm==26.10.*" not in updated


def test_unit_version_override_wins_over_family_fallback(tmp_path: Path) -> None:
    result = _run_helper(
        tmp_path,
        {
            "package_versions": {"wheel": "26.10.0"},
            "unit_package_versions": {"wheel:rmm": "26.12.0"},
        },
    )

    assert result.returncode == 0, result.stderr
    updated = (tmp_path / "python/rmm/pyproject.toml").read_text(encoding="utf-8")
    assert updated.count("librmm==26.12.0") == 2


def test_family_version_is_used_when_unit_override_is_absent(tmp_path: Path) -> None:
    result = _run_helper(
        tmp_path,
        {
            "package_versions": {"wheel": "26.12.0"},
            "unit_package_versions": {},
        },
    )

    assert result.returncode == 0, result.stderr
    updated = (tmp_path / "python/rmm/pyproject.toml").read_text(encoding="utf-8")
    assert updated.count("librmm==26.12.0") == 2


def test_missing_candidate_metadata_preserves_production_metadata(tmp_path: Path) -> None:
    result = _run_helper(tmp_path, None)

    assert result.returncode == 0
    updated = (tmp_path / "python/rmm/pyproject.toml").read_text(encoding="utf-8")
    assert updated.count("librmm==26.10.*") == 2


@pytest.mark.parametrize(
    "versions",
    [
        {"package_versions": {}, "unit_package_versions": {}},
        {"package_versions": {"wheel": "26.12"}},
        {"package_versions": {"wheel": "26.12.0+local"}},
    ],
)
def test_invalid_or_missing_frozen_version_fails_without_edit(
    tmp_path: Path, versions: dict[str, object]
) -> None:
    result = _run_helper(tmp_path, versions)

    assert result.returncode == 1
    updated = (tmp_path / "python/rmm/pyproject.toml").read_text(encoding="utf-8")
    assert updated.count("librmm==26.10.*") == 2


def test_unexpected_generated_constraint_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "python/rmm/pyproject.toml"
    _write_pyproject(target)
    target.write_text(
        target.read_text(encoding="utf-8").replace("librmm==26.10.*", "librmm>=26.10"),
        encoding="utf-8",
    )
    helper = tmp_path / "ci/prepare_candidate_wheel_dependencies.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HELPER, helper)
    environment = os.environ.copy()
    environment["RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS"] = json.dumps(
        {"package_versions": {"wheel": "26.12.0"}}
    )

    result = subprocess.run(
        [sys.executable, str(helper)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "expected exactly one generated wildcard" in result.stderr
    assert "librmm>=26.10" in target.read_text(encoding="utf-8")
