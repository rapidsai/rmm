# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from typing import Optional, cast

from rmm.pylibrmm.stream import Stream

class CudaStreamFlags(IntEnum):
    SYNC_DEFAULT = cast(int, ...)
    NON_BLOCKING = cast(int, ...)

class CudaStreamPool:
    def __init__(
        self, pool_size: int = 16, flags: CudaStreamFlags = CudaStreamFlags.SYNC_DEFAULT
    ): ...
    def get_stream(self, stream_id: Optional[int] = None) -> Stream: ...
    def get_pool_size(self) -> int: ...
