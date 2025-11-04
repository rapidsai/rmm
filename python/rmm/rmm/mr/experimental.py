# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Experimental memory resource features that may change or be removed in future releases."""

from rmm.pylibrmm.memory_resource.experimental import (
    CudaAsyncManagedMemoryResource,
)

__all__ = [
    "CudaAsyncManagedMemoryResource",
]
