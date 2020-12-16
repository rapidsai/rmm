/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#pragma once

#include <cstddef>
#include <utility>
#include <rmm/mr/memory_resource.hpp>

namespace rmm {
namespace mr {

/**---------------------------------------------------------------------------*
 * @brief Base class for host memory allocation.
 *
 * This is based on `std::pmr::memory_resource`:
 * https://en.cppreference.com/w/cpp/memory/memory_resource
 *
 * When C++17 is available for use in RMM, `rmm::host_memory_resource` should
 * inherit from `std::pmr::memory_resource`.
 *
 * This class serves as the interface that all host memory resource
 * implementations must satisfy.
 *
 * There are two private, pure virtual functions that all derived classes must
 * implement: `do_allocate` and `do_deallocate`. Optionally, derived classes may
 * also override `is_equal`. By default, `is_equal` simply performs an identity
 * comparison.
 *
 * The public, non-virtual functions `allocate`, `deallocate`, and `is_equal`
 * simply call the private virtual functions. The reason for this is to allow
 * implementing shared, default behavior in the base class. For example, the
 * base class' `allocate` function may log every allocation, no matter what
 * derived class implementation is used.
 *
 *---------------------------------------------------------------------------**/
using host_memory_resource = memory_resource<memory_kind::host, allocation_order::host>;

}  // namespace mr
}  // namespace rmm
