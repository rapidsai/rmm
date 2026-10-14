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
_SOURCE_VERSION = (
    (REPOSITORY_ROOT / "VERSION").read_text(encoding="utf-8").strip()
)
_SOURCE_MAJOR, _SOURCE_MINOR, *_ = _SOURCE_VERSION.split(".")
_SOURCE_WILDCARD = f"{_SOURCE_MAJOR}.{_SOURCE_MINOR}.*"
_SOURCE_REQUIREMENT = f"librmm=={_SOURCE_WILDCARD},>=0.0.0a0"
_SOURCE_REQUIREMENTS = {
    package: f"{package}=={_SOURCE_WILDCARD},>=0.0.0a0"
    for package in ("librmm", "librmm-cu12", "librmm-cu13", "librmm-tests")
}
_FROZEN_VERSION = f"{_SOURCE_MAJOR}.{int(_SOURCE_MINOR) + 2}.0"


def _write_pyproject(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""[build-system]
requires = ["rapids-build-backend>=0.4.0,<0.5.0"]

[project]
dependencies = ["{_SOURCE_REQUIREMENT}"]

[tool.rapids-build-backend]
requires = ["{_SOURCE_REQUIREMENT}"]
""",
        encoding="utf-8",
    )


def _write_dependencies(path: Path) -> None:
    path.write_text(
        f"""includes:
  - depends_on_librmm
depends_on_librmm:
  common:
    - output_types: conda
      packages:
        - {_SOURCE_REQUIREMENTS["librmm"]}
  specific:
    - output_types: [requirements, pyproject]
      packages:
        - {_SOURCE_REQUIREMENTS["librmm-cu12"]}
    - output_types: [requirements, pyproject]
      packages:
        - {_SOURCE_REQUIREMENTS["librmm-cu13"]}
depends_on_librmm_tests:
  common:
    - output_types: conda
      packages:
        - {_SOURCE_REQUIREMENTS["librmm-tests"]}
""",
        encoding="utf-8",
    )


def _run_helper(
    root: Path,
    versions: dict[str, object] | None,
    *,
    candidate_mode: bool = True,
) -> subprocess.CompletedProcess[str]:
    target = root / "python/rmm/pyproject.toml"
    _write_pyproject(target)
    _write_dependencies(root / "dependencies.yaml")
    helper = root / "ci/prepare_candidate_wheel_dependencies.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HELPER, helper)
    environment = os.environ.copy()
    environment.pop("RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS", None)
    environment["RAPIDS_RELEASE_CANDIDATE"] = str(candidate_mode).lower()
    if versions is not None:
        environment["RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS"] = json.dumps(
            versions
        )
    return subprocess.run(
        [sys.executable, str(helper)],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_candidate_rewrites_runtime_and_build_backend_requirements(
    tmp_path: Path,
) -> None:
    result = _run_helper(
        tmp_path,
        {
            "package_versions": {"wheel": _FROZEN_VERSION},
            "unit_package_versions": {"wheel:rmm": _FROZEN_VERSION},
        },
    )

    assert result.returncode == 0, result.stderr
    updated = (tmp_path / "python/rmm/pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert updated.count(f"librmm=={_FROZEN_VERSION}") == 2
    assert _SOURCE_REQUIREMENT not in updated
    dependency_file = (tmp_path / "dependencies.yaml").read_text(
        encoding="utf-8"
    )
    assert dependency_file.count(f"=={_FROZEN_VERSION},") == 3
    assert _SOURCE_REQUIREMENTS["librmm-tests"] in dependency_file


def test_unit_version_override_wins_over_family_fallback(
    tmp_path: Path,
) -> None:
    result = _run_helper(
        tmp_path,
        {
            "package_versions": {"wheel": _SOURCE_VERSION},
            "unit_package_versions": {"wheel:rmm": _FROZEN_VERSION},
        },
    )

    assert result.returncode == 0, result.stderr
    updated = (tmp_path / "python/rmm/pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert updated.count(f"librmm=={_FROZEN_VERSION}") == 2
    dependency_file = (tmp_path / "dependencies.yaml").read_text(
        encoding="utf-8"
    )
    assert dependency_file.count(f"=={_FROZEN_VERSION},") == 3


def test_family_version_is_used_when_unit_override_is_absent(
    tmp_path: Path,
) -> None:
    result = _run_helper(
        tmp_path,
        {
            "package_versions": {"wheel": _FROZEN_VERSION},
            "unit_package_versions": {},
        },
    )

    assert result.returncode == 0, result.stderr
    updated = (tmp_path / "python/rmm/pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert updated.count(f"librmm=={_FROZEN_VERSION}") == 2
    dependency_file = (tmp_path / "dependencies.yaml").read_text(
        encoding="utf-8"
    )
    assert dependency_file.count(f"=={_FROZEN_VERSION},") == 3


def test_missing_candidate_metadata_preserves_production_metadata(
    tmp_path: Path,
) -> None:
    result = _run_helper(tmp_path, None, candidate_mode=False)

    assert result.returncode == 0
    updated = (tmp_path / "python/rmm/pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert updated.count(_SOURCE_REQUIREMENT) == 2
    dependency_file = (tmp_path / "dependencies.yaml").read_text(
        encoding="utf-8"
    )
    assert (
        sum(
            dependency_file.count(requirement)
            for requirement in _SOURCE_REQUIREMENTS.values()
        )
        == 4
    )


def test_candidate_mode_without_version_map_fails_without_edit(
    tmp_path: Path,
) -> None:
    result = _run_helper(tmp_path, None)

    assert result.returncode == 1
    assert "candidate unit-version metadata is required" in result.stderr
    updated = (tmp_path / "python/rmm/pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert updated.count(_SOURCE_REQUIREMENT) == 2
    dependency_file = (tmp_path / "dependencies.yaml").read_text(
        encoding="utf-8"
    )
    assert (
        sum(
            dependency_file.count(requirement)
            for requirement in _SOURCE_REQUIREMENTS.values()
        )
        == 4
    )


@pytest.mark.parametrize(
    "versions",
    [
        {"package_versions": {}, "unit_package_versions": {}},
        {"package_versions": {"wheel": _FROZEN_VERSION.rsplit(".", 1)[0]}},
        {"package_versions": {"wheel": f"{_FROZEN_VERSION}+local"}},
    ],
)
def test_invalid_or_missing_frozen_version_fails_without_edit(
    tmp_path: Path, versions: dict[str, object]
) -> None:
    result = _run_helper(tmp_path, versions)

    assert result.returncode == 1
    updated = (tmp_path / "python/rmm/pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert updated.count(_SOURCE_REQUIREMENT) == 2
    dependency_file = (tmp_path / "dependencies.yaml").read_text(
        encoding="utf-8"
    )
    assert (
        sum(
            dependency_file.count(requirement)
            for requirement in _SOURCE_REQUIREMENTS.values()
        )
        == 4
    )


def test_unexpected_generated_constraint_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "python/rmm/pyproject.toml"
    _write_pyproject(target)
    _write_dependencies(tmp_path / "dependencies.yaml")
    invalid_requirement = _SOURCE_REQUIREMENT.split(",", 1)[0].replace(
        "==", ">="
    )
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            _SOURCE_REQUIREMENT, invalid_requirement
        ),
        encoding="utf-8",
    )
    helper = tmp_path / "ci/prepare_candidate_wheel_dependencies.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HELPER, helper)
    environment = os.environ.copy()
    environment["RAPIDS_RELEASE_CANDIDATE"] = "true"
    environment["RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS"] = json.dumps(
        {"package_versions": {"wheel": _FROZEN_VERSION}}
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
    assert invalid_requirement in target.read_text(encoding="utf-8")


def test_invalid_dependency_file_does_not_partially_rewrite_pyproject(
    tmp_path: Path,
) -> None:
    pyproject = tmp_path / "python/rmm/pyproject.toml"
    _write_pyproject(pyproject)
    dependency_file = tmp_path / "dependencies.yaml"
    _write_dependencies(dependency_file)
    dependency_file.write_text(
        dependency_file.read_text(encoding="utf-8").replace(
            _SOURCE_REQUIREMENTS["librmm-cu13"],
            f"librmm-cu13>={_SOURCE_WILDCARD},>=0.0.0a0",
        ),
        encoding="utf-8",
    )
    helper = tmp_path / "ci/prepare_candidate_wheel_dependencies.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HELPER, helper)
    environment = os.environ.copy()
    environment["RAPIDS_RELEASE_CANDIDATE"] = "true"
    environment["RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS"] = json.dumps(
        {"unit_package_versions": {"wheel:rmm": _FROZEN_VERSION}}
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
    assert "expected one generated librmm" in result.stderr
    assert (
        pyproject.read_text(encoding="utf-8").count(_SOURCE_REQUIREMENT) == 2
    )
