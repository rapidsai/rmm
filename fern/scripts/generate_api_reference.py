#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Generate Fern API reference pages from Sphinx-rendered Markdown."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FERN_ROOT = REPO_ROOT / "fern"
SPHINX_ROOT = FERN_ROOT / "sphinx"
SPHINX_SOURCE_DIR = SPHINX_ROOT / "source"
SPHINX_BUILD_DIR = SPHINX_ROOT / "build"
MARKDOWN_BUILD_DIR = SPHINX_BUILD_DIR / "markdown"
API_OUTPUT_DIR = FERN_ROOT / "pages" / "api_reference"
DOXYGEN_DIR = REPO_ROOT / "cpp" / "doxygen"
DOXYGEN_XML_INDEX = DOXYGEN_DIR / "xml" / "index.xml"

GENERATED_NOTICE = (
    "<!-- Generated from the Sphinx API extraction build. "
    "Do not edit directly. -->\n\n"
)
API_SOURCE_DIRS = ("cpp", "python")


def relative_to_repo(path: Path) -> str:
    """Return a repo-relative path for concise command output."""
    return str(path.relative_to(REPO_ROOT))


def require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise SystemExit(
            f"{command!r} is required to generate Fern API reference pages. "
            "Install the docs environment from dependencies.yaml."
        )


def docs_environment() -> dict[str, str]:
    env = os.environ.copy()
    version = (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    major_minor = ".".join(version.split(".")[:2])
    env.setdefault("RAPIDS_VERSION", version)
    env.setdefault("RAPIDS_VERSION_MAJOR_MINOR", major_minor)
    return env


def run_command(args: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    display = " ".join(args)
    print(f"+ ({relative_to_repo(cwd)}) {display}", file=sys.stderr)
    subprocess.run(args, cwd=cwd, env=env, check=True)


def build_doxygen_xml(env: dict[str, str]) -> None:
    require_command("doxygen")
    run_command(["doxygen", "Doxyfile"], cwd=DOXYGEN_DIR, env=env)
    if not DOXYGEN_XML_INDEX.exists():
        raise SystemExit(
            "Doxygen completed without producing cpp/doxygen/xml/index.xml"
        )


def build_sphinx_markdown(env: dict[str, str]) -> None:
    require_command("sphinx-build")
    shutil.rmtree(MARKDOWN_BUILD_DIR, ignore_errors=True)
    run_command(
        [
            "sphinx-build",
            "-M",
            "markdown",
            ".",
            "../build",
        ],
        cwd=SPHINX_SOURCE_DIR,
        env=env,
    )
    if not MARKDOWN_BUILD_DIR.exists():
        raise SystemExit(
            "Sphinx completed without producing fern/sphinx/build/markdown"
        )


def normalize_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    return f"{GENERATED_NOTICE}{text}\n"


def copy_generated_markdown_pages(markdown_dir: Path, output_dir: Path) -> int:
    shutil.rmtree(output_dir, ignore_errors=True)
    copied = 0
    for source_dir_name in API_SOURCE_DIRS:
        source_dir = markdown_dir / source_dir_name
        if not source_dir.exists():
            continue
        for source_path in sorted(source_dir.rglob("*.md")):
            target_path = output_dir / source_path.relative_to(markdown_dir)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(
                normalize_markdown(source_path.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
            copied += 1
    return copied


def main() -> int:
    env = docs_environment()
    build_doxygen_xml(env)
    build_sphinx_markdown(env)
    copied = copy_generated_markdown_pages(MARKDOWN_BUILD_DIR, API_OUTPUT_DIR)
    if copied == 0:
        raise SystemExit("No API reference Markdown pages were generated")
    print(f"Generated {copied} Fern API reference pages.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
