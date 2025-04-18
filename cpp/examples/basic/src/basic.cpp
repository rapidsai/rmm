/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>

int main(int argc, char** argv)
{
  // Construct a CUDA async memory resource using RAPIDS Memory Manager (RMM).
  // This uses a memory pool managed by the CUDA driver, using half of the
  // available GPU memory.
  rmm::mr::cuda_async_memory_resource mr{rmm::percent_of_free_device_memory(50)};

  // Create a CUDA stream for asynchronous allocations
  auto stream = rmm::cuda_stream{};

  // Create a device_uvector with this stream and memory resource
  auto const size{12345};
  rmm::device_uvector<int> vec(size, stream, mr);
  std::cout << "vec size: " << vec.size() << std::endl;

  // Synchronize the stream
  stream.synchronize();

  return 0;
}
