/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <utility>

namespace rmm {

cuda_device_id get_current_cuda_device()
{
  cuda_device_id::value_type dev_id{-1};
  RMM_ASSERT_CUDA_SUCCESS(cudaGetDevice(&dev_id));
  return cuda_device_id{dev_id};
}

int get_num_cuda_devices()
{
  cuda_device_id::value_type num_dev{-1};
  RMM_ASSERT_CUDA_SUCCESS(cudaGetDeviceCount(&num_dev));
  return num_dev;
}

std::pair<std::size_t, std::size_t> available_device_memory()
{
  std::size_t free{};
  std::size_t total{};
  RMM_CUDA_TRY(cudaMemGetInfo(&free, &total));
  return {free, total};
}

std::size_t percent_of_free_device_memory(int percent)
{
  [[maybe_unused]] auto const [free, total] = rmm::available_device_memory();
  auto fraction                             = static_cast<double>(percent) / 100.0;
  return rmm::align_down(static_cast<std::size_t>(static_cast<double>(free) * fraction),
                         rmm::CUDA_ALLOCATION_ALIGNMENT);
}

cuda_set_device_raii::cuda_set_device_raii(cuda_device_id dev_id)
  : old_device_{get_current_cuda_device()},
    needs_reset_{dev_id.value() >= 0 && old_device_ != dev_id}
{
  if (needs_reset_) { RMM_ASSERT_CUDA_SUCCESS(cudaSetDevice(dev_id.value())); }
}

cuda_set_device_raii::~cuda_set_device_raii() noexcept
{
  if (needs_reset_) { RMM_ASSERT_CUDA_SUCCESS(cudaSetDevice(old_device_.value())); }
}

}  // namespace rmm
