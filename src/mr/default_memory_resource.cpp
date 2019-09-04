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
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/cnmem_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/default_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <atomic>

namespace rmm {
namespace mr {

device_memory_resource* initial_resource() {
  static cuda_memory_resource resource{};
  return &resource;
}

device_memory_resource* pool_resource(std::size_t pool_size){
   static cnmem_memory_resource mr{pool_size};
   return &mr;
}


device_memory_resource* managed_resource(){
   static managed_memory_resource mr{};
   return &mr;
}

namespace {





// Use an atomic to guarantee thread safety
std::atomic<device_memory_resource*>& get_default() {
  static std::atomic<device_memory_resource*> res{initial_resource()};
  return res;
}
}  // namespace

device_memory_resource* get_default_resource() { return get_default().load(); }

device_memory_resource* set_default_resource(
    device_memory_resource* new_resource) {
  new_resource = (new_resource == nullptr) ? initial_resource() : new_resource;
  return get_default().exchange(new_resource);
}
}  // namespace mr
}  // namespace rmm
