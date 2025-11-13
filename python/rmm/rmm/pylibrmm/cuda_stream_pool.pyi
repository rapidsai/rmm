# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from rmm.pylibrmm.cuda_stream import CudaStreamFlags
from rmm.pylibrmm.stream import Stream

class CudaStreamPool:
    def __init__(
        self,
        pool_size: int = 16,
        flags: CudaStreamFlags = CudaStreamFlags.SYNC_DEFAULT,
    ) -> None: ...
    def get_stream(self, stream_id: Optional[int] = None) -> Stream: ...
    def get_pool_size(self) -> int: ...
