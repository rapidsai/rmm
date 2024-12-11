# Copyright (c) 2024, NVIDIA CORPORATION.
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

from rmm.librmm.memory_resource cimport (
    CppExcept,
    allocate_callback_t,
    allocation_handle_type,
    available_device_memory,
    binning_memory_resource,
    callback_memory_resource,
    cuda_async_memory_resource,
    cuda_memory_resource,
    deallocate_callback_t,
    device_memory_resource,
    failure_callback_resource_adaptor,
    failure_callback_t,
    fixed_size_memory_resource,
    limiting_resource_adaptor,
    logging_resource_adaptor,
    managed_memory_resource,
    percent_of_free_device_memory,
    pool_memory_resource,
    prefetch_resource_adaptor,
    sam_headroom_memory_resource,
    statistics_resource_adaptor,
    system_memory_resource,
    throw_cpp_except,
    tracking_resource_adaptor,
    translate_python_except_to_cpp,
)
from rmm.pylibrmm.memory_resource cimport (
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
    get_current_device_resource,
)
