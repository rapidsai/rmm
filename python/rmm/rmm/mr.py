# Copyright (c) 2020-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from rmm.pylibrmm.memory_resource import (
    ArenaMemoryResource,
    BinningMemoryResource,
    CallbackMemoryResource,
    CudaAsyncMemoryResource,
    CudaMemoryResource,
    DeviceMemoryResource,
    FailureCallbackResourceAdaptor,
    FixedSizeMemoryResource,
    LimitingResourceAdaptor,
    LoggingResourceAdaptor,
    ManagedMemoryResource,
    PoolMemoryResource,
    PrefetchResourceAdaptor,
    SamHeadroomMemoryResource,
    StatisticsResourceAdaptor,
    SystemMemoryResource,
    TrackingResourceAdaptor,
    UpstreamResourceAdaptor,
    _flush_logs,
    _initialize,
    available_device_memory,
    disable_logging,
    enable_logging,
    get_current_device_resource,
    get_current_device_resource_type,
    get_log_filenames,
    get_per_device_resource,
    get_per_device_resource_type,
    is_initialized,
    set_current_device_resource,
    set_per_device_resource,
)

__all__ = [
    "ArenaMemoryResource",
    "BinningMemoryResource",
    "CallbackMemoryResource",
    "CudaAsyncMemoryResource",
    "CudaMemoryResource",
    "DeviceMemoryResource",
    "FailureCallbackResourceAdaptor",
    "FixedSizeMemoryResource",
    "LimitingResourceAdaptor",
    "LoggingResourceAdaptor",
    "ManagedMemoryResource",
    "PoolMemoryResource",
    "PrefetchResourceAdaptor",
    "SamHeadroomMemoryResource",
    "StatisticsResourceAdaptor",
    "SystemMemoryResource",
    "TrackingResourceAdaptor",
    "UpstreamResourceAdaptor",
    "_flush_logs",
    "_initialize",
    "available_device_memory",
    "disable_logging",
    "enable_logging",
    "get_current_device_resource",
    "get_current_device_resource_type",
    "get_log_filenames",
    "get_per_device_resource",
    "get_per_device_resource_type",
    "is_initialized",
    "set_current_device_resource",
    "set_per_device_resource",
]
