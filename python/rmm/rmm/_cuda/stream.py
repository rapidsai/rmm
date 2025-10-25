# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

from rmm.pylibrmm.stream import (
    DEFAULT_STREAM,
    LEGACY_DEFAULT_STREAM,
    PER_THREAD_DEFAULT_STREAM,
    Stream,
)

__all__ = [
    "DEFAULT_STREAM",
    "LEGACY_DEFAULT_STREAM",
    "PER_THREAD_DEFAULT_STREAM",
    "Stream",
]

warnings.warn(
    "The `rmm._cuda.stream` module is deprecated in 25.02 and will be removed in a future release. Use `rmm.pylibrmm.stream` instead.",
    FutureWarning,
    stacklevel=2,
)
