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
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/logger.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuda_runtime.h>

#include <iostream>

// CUDA kernel to perform addition with potential out-of-bounds access
__global__ void add_with_oob(int* a, int* b, int* c) { *c = *a + *b; }

int main(int argc, char** argv)
{
  rmm::default_logger().set_level(rapids_logger::level_enum::debug);

  // Create a CUDA memory resource to use as the upstream resource
  rmm::mr::cuda_memory_resource upstream_mr;

  // // Create a pool memory resource with initial size of 1GB and maximum size of 2GB
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(&upstream_mr, 1024, 1024);

  auto& mr = pool_mr;
  // auto& mr = upstream_mr;

  // Create a CUDA stream for asynchronous operations
  auto stream = rmm::cuda_stream_default;

  // Allocate memory for 2 integers
  void* a = mr.allocate(sizeof(int), stream);
  void* b = mr.allocate(sizeof(int), stream);

  // Initialize values
  int h_a = 100;
  int h_b = 101;

  // Copy values to device
  cudaMemcpyAsync(a, &h_a, sizeof(int), cudaMemcpyDefault, stream);
  cudaMemcpyAsync(b, &h_b, sizeof(int), cudaMemcpyDefault, stream);

  void* c = reinterpret_cast<char*>(b) + sizeof(int);

  // Launch kernel with potential out-of-bounds access
  add_with_oob<<<1, 1, 0, stream>>>(
    reinterpret_cast<int*>(a), reinterpret_cast<int*>(b), reinterpret_cast<int*>(c));

  int h_c;
  cudaMemcpyAsync(&h_c, c, sizeof(int), cudaMemcpyDefault, stream);

  // Synchronize the stream
  stream.synchronize();

  // Check for CUDA errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
  }

  std::cout << "c: " << h_c << std::endl;

  return 0;
}