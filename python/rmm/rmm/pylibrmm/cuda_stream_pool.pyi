# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from rmm.pylibrmm.stream import CudaStreamFlags, Stream

class CudaStreamPool:
    def __init__(
        self,
        pool_size: int = ...,
        flags: CudaStreamFlags = ...,
    ) -> None: ...
    def get_stream(self, stream_id: Optional[int] = ...) -> Stream: ...
    def get_pool_size(self) -> int: ...
