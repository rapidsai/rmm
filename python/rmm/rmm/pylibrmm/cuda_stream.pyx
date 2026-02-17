# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

# Re-export from stream for backward compat
# prefer from rmm.pylibrmm.stream import CudaStream, CudaStreamFlags
from rmm.pylibrmm.stream import (
    CudaStream,
    CudaStreamFlags,
)

warnings.warn(
    "rmm.pylibrmm.cuda_stream is deprecated; use rmm.pylibrmm.stream for "
    "CudaStream and CudaStreamFlags.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CudaStream",
    "CudaStreamFlags",
]
