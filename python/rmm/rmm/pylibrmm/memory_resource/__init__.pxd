# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource._memory_resource cimport (
    ArenaMemoryResource,
    BinningMemoryResource,
    CallbackMemoryResource,
    CudaAsyncMemoryResource,
    CudaAsyncViewMemoryResource,
    CudaMemoryResource,
    DeviceMemoryResource,
    FailureCallbackResourceAdaptor,
    FixedSizeMemoryResource,
    LimitingResourceAdaptor,
    LoggingResourceAdaptor,
    ManagedMemoryResource,
    PinnedHostMemoryResource,
    PoolMemoryResource,
    PrefetchResourceAdaptor,
    SamHeadroomMemoryResource,
    StatisticsResourceAdaptor,
    SystemMemoryResource,
    TrackingResourceAdaptor,
    UpstreamResourceAdaptor,
    get_current_device_resource,
)
