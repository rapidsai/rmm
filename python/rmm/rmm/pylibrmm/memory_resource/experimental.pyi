# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource._memory_resource import DeviceMemoryResource

class CudaAsyncManagedMemoryResource(DeviceMemoryResource):
    def __init__(self) -> None: ...
    def pool_handle(self) -> int: ...
