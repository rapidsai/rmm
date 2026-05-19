# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_build_docs_pins_npx_fern_api_fallback():
    script = REPO_ROOT / "fern" / "build_docs.sh"
    text = script.read_text(encoding="utf-8")

    assert '"fern-api@5.30.4"' in text
    assert 'FERN_CMD=("npx" "--yes" "fern-api")' not in text
