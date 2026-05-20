# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FERN_ROOT = REPO_ROOT / "fern"


def test_build_docs_pins_npx_fern_api_fallback():
    script = REPO_ROOT / "fern" / "build_docs.sh"
    text = script.read_text(encoding="utf-8")

    assert '"fern-api@5.30.4"' in text
    assert 'FERN_CMD=("npx" "--yes" "fern-api")' not in text


def test_build_docs_does_not_run_handwritten_api_generator():
    script = REPO_ROOT / "fern" / "build_docs.sh"
    text = script.read_text(encoding="utf-8")

    assert "generate_api_reference.py" not in text
    assert "generate_api_reference" not in text


def test_fern_docs_do_not_link_to_legacy_api_reference():
    docs_yml = FERN_ROOT / "docs.yml"
    page_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((FERN_ROOT / "pages").rglob("*.md"))
    )

    assert "docs.rapids.ai/api/rmm" not in page_text
    assert "api_reference" not in docs_yml.read_text(encoding="utf-8")
