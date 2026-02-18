# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# backwards compat file reexporting CudaStream and CudaStreamFlags

import warnings

from rmm.pylibrmm.stream cimport _OwningStream as CudaStream

import sys

from rmm.pylibrmm.stream import CudaStreamFlags, _OwningStream

sys.modules[__name__].CudaStream = _OwningStream

warnings.warn(
    "rmm.pylibrmm.cuda_stream is deprecated; use rmm.pylibrmm.stream for "
    "CudaStreamFlags.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CudaStream",
    "CudaStreamFlags",
]
