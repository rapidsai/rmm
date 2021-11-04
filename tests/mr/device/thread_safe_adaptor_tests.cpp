/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/thread_safe_resource_adaptor.hpp>

// explicit instantiation for code coverage testing. Ensures unused template class methods are
// included in coverage analysis.
template class rmm::mr::thread_safe_resource_adaptor<rmm::mr::device_memory_resource>;
