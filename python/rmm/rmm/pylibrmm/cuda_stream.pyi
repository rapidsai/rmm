# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

class CudaStreamFlags(IntEnum):
    SYNC_DEFAULT = ...
    NON_BLOCKING = ...

class CudaStream:
    def __init__(self) -> None: ...
