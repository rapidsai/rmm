# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import rmm


def test_version_constants_are_populated() -> None:
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(rmm.__git_commit__, str)

    # __version__ should always be non-empty
    assert isinstance(rmm.__version__, str)
    assert len(rmm.__version__) > 0
