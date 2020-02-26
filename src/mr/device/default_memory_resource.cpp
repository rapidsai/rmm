/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <rmm/mr/device/cnmem_managed_memory_resource.hpp>
#include <rmm/mr/device/cnmem_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/fixed_multisize_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/hybrid_memory_resource.hpp>

#include <atomic>

namespace rmm {
namespace mr {

// nested MR type names can get long...
using pool_mr = pool_memory_resource<cuda_memory_resource>;
using fixed_multisize_mr = fixed_multisize_memory_resource<pool_mr>;
using hybrid_mr = hybrid_memory_resource<fixed_multisize_mr, pool_mr>;

namespace detail{

// gets the default memory_resource when none is set
device_memory_resource* initial_resource() {
  static cuda_memory_resource cuda_res{};
#if 1
  static pool_mr pool_res{&cuda_res};
  static hybrid_mr hybrid_res(new fixed_multisize_mr(&pool_res), &pool_res);
  return &hybrid_res;
#else
  return &cuda_res;
#endif
}
} // namespace detail

namespace {

// Use an atomic to guarantee thread safety
std::atomic<device_memory_resource*>& get_default() {
  static std::atomic<device_memory_resource*> res{detail::initial_resource()};
  return res;
}

}  // namespace anonymous

device_memory_resource* get_default_resource() {
  return get_default().load();
}

device_memory_resource*
set_default_resource(device_memory_resource* new_resource) {
  new_resource = (new_resource == nullptr) ? detail::initial_resource() 
                                           : new_resource;
  return get_default().exchange(new_resource);
}

}  // namespace mr
}  // namespace rmm
