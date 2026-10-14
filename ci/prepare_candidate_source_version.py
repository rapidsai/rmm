# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the disposable RMM source tree for its frozen candidate version."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_VERSION_PATTERN = re.compile(r"\d{2}\.\d{2}\.\d{2}")


def _candidate_conda_version(raw_versions: str) -> str:
    """Resolve RMM's frozen Conda version from validated train metadata."""
    try:
        versions: Any = json.loads(raw_versions)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "candidate unit-version metadata is not valid JSON"
        ) from exc

    if not isinstance(versions, dict):
        raise ValueError("candidate unit-version metadata must be an object")

    unit_versions = versions.get("unit_package_versions", {})
    family_versions = versions.get("package_versions", {})
    if not isinstance(unit_versions, dict) or not isinstance(
        family_versions, dict
    ):
        raise ValueError("candidate version maps must be objects")

    version = unit_versions.get("conda:rmm", family_versions.get("conda"))
    if (
        not isinstance(version, str)
        or _VERSION_PATTERN.fullmatch(version) is None
    ):
        raise ValueError("frozen conda:rmm version is missing or invalid")
    return version


def main() -> int:
    """Write VERSION only for candidate builds; preserve ordinary checkout state."""
    candidate_mode = os.environ.get(
        "RAPIDS_RELEASE_CANDIDATE", "false"
    ).lower()
    if candidate_mode not in {"true", "false"}:
        print(
            "RAPIDS_RELEASE_CANDIDATE must be true or false", file=sys.stderr
        )
        return 1
    if candidate_mode == "false":
        return 0

    try:
        raw_versions = os.environ.get("RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS")
        if raw_versions is None:
            raise ValueError("candidate unit-version metadata is required")
        version = _candidate_conda_version(raw_versions)
        Path("VERSION").write_text(f"{version}\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(
            f"Unable to prepare frozen RMM candidate source version: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"Prepared frozen candidate RMM source version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
