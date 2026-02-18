# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

# Re-export from stream for backward compat.
from rmm.pylibrmm.stream import CudaStreamFlags

warnings.warn(
    "rmm.pylibrmm.cuda_stream is deprecated; use rmm.pylibrmm.stream for "
    "CudaStreamFlags.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CudaStreamFlags",
]
