# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Align RMM's generated wheel requirements with the frozen candidate train."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_VERSION_PATTERN = re.compile(r"\d+\.\d+\.\d+")
_LIBRMM_REQUIREMENT = re.compile(r'librmm==\d+\.\d+\.\*,>=0\.0\.0a0')


def _candidate_version(raw_versions: str) -> str:
    """Return the frozen wheel version for RMM, preferring its unit override."""
    try:
        versions: Any = json.loads(raw_versions)
    except json.JSONDecodeError as exc:
        raise ValueError("candidate unit-version metadata is not valid JSON") from exc

    if not isinstance(versions, dict):
        raise ValueError("candidate unit-version metadata must be an object")

    unit_versions = versions.get("unit_package_versions", {})
    family_versions = versions.get("package_versions", {})
    if not isinstance(unit_versions, dict) or not isinstance(family_versions, dict):
        raise ValueError("candidate version maps must be objects")

    version = unit_versions.get("wheel:rmm", family_versions.get("wheel"))
    if not isinstance(version, str) or _VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError("frozen wheel:rmm version is missing or invalid")
    return version


def _rewrite_pyproject(path: Path, version: str) -> None:
    """Rewrite only the generated runtime and backend librmm constraints."""
    import tomllib

    original = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(original)
    requirement_lists = (
        parsed.get("project", {}).get("dependencies", []),
        parsed.get("tool", {}).get("rapids-build-backend", {}).get("requires", []),
    )
    selected = [
        requirement
        for requirements in requirement_lists
        for requirement in requirements
        if isinstance(requirement, str) and requirement.startswith("librmm==")
    ]
    if len(selected) != 2 or not all(
        _LIBRMM_REQUIREMENT.fullmatch(requirement) for requirement in selected
    ):
        raise ValueError(
            "expected exactly one generated wildcard librmm constraint in "
            "project.dependencies and tool.rapids-build-backend.requires"
        )

    updated, count = _LIBRMM_REQUIREMENT.subn(f"librmm=={version}", original)
    if count != 2:
        raise ValueError("expected exactly two generated librmm constraints")
    if updated != original:
        path.write_text(updated, encoding="utf-8")


def main() -> int:
    """Update candidate metadata; leave ordinary production builds untouched."""
    raw_versions = os.environ.get("RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS")
    if raw_versions is None:
        return 0

    try:
        version = _candidate_version(raw_versions)
        _rewrite_pyproject(Path("python/rmm/pyproject.toml"), version)
    except (OSError, ValueError) as exc:
        print(f"Unable to prepare frozen RMM candidate requirements: {exc}", file=sys.stderr)
        return 1

    print(f"Applied frozen candidate librmm=={version} to RMM wheel metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
