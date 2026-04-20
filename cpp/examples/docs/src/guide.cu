/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Code examples for docs/user_guide/guide.md
//
// Include directives that appear inside function bodies are intentional:
// they are no-ops (headers use #pragma once) and exist so that
// literalinclude snippets display the includes alongside the code.

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/logging_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>

__global__ void trivial_kernel(int* data) { data[0] = 42; }

void explicit_resource()
{
  // clang-format off
  // [explicit-resource]
  rmm::mr::cuda_async_memory_resource async_mr;
  rmm::cuda_stream stream;

  // Pass the resource explicitly
  rmm::device_buffer buffer(1024, stream.view(), async_mr);
  // [/explicit-resource]
  // clang-format on

  assert(buffer.size() == 1024);
}

void current_device_resource()
{
  // clang-format off
  // [current-device-resource]
  #include <rmm/mr/cuda_async_memory_resource.hpp>
  #include <rmm/mr/per_device_resource.hpp>

  rmm::mr::cuda_async_memory_resource async_mr;
  rmm::mr::set_current_device_resource_ref(async_mr);

  // Allocations that don't specify a resource use the current device resource
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref();
  // [/current-device-resource]
  // clang-format on

  (void)mr;
}

void device_buffer_example()
{
  // clang-format off
  // [device-buffer]
  #include <rmm/device_buffer.hpp>

  rmm::cuda_stream stream;

  // Allocate 1024 bytes
  rmm::device_buffer buffer(1024, stream.view());

  // Access pointer and size
  void* ptr = buffer.data();
  std::size_t size = buffer.size();

  // Resize (may reallocate)
  buffer.resize(2048, stream.view());

  // Copy construct (deep copy)
  rmm::device_buffer buffer2(buffer, stream.view());
  // [/device-buffer]
  // clang-format on

  assert(buffer.size() == 2048);
  assert(buffer2.size() == 2048);
  (void)ptr;
  (void)size;
}

void device_uvector_example()
{
  // clang-format off
  // [device-uvector]
  #include <rmm/device_uvector.hpp>
  #include <rmm/exec_policy.hpp>
  #include <thrust/fill.h>

  rmm::cuda_stream stream;

  // Allocate 100 elements
  rmm::device_uvector<int> vec(100, stream.view());

  // Access as pointer
  int* ptr = vec.data();

  // Access as iterators
  auto begin = vec.begin();
  auto end = vec.end();

  // Initialize with Thrust
  thrust::fill(rmm::exec_policy(stream.view()), vec.begin(), vec.end(), 42);

  // Resize
  vec.resize(200, stream.view());
  // [/device-uvector]
  // clang-format on

  assert(vec.size() == 200);
  (void)ptr;
  (void)begin;
  (void)end;
}

void device_scalar_example()
{
  // clang-format off
  // [device-scalar]
  #include <rmm/device_scalar.hpp>

  rmm::cuda_stream stream;

  // Allocate single int
  rmm::device_scalar<int> scalar(stream.view());

  // Set value from host (async on stream)
  scalar.set_value(42, stream.view());

  // Get value to host (async on stream)
  int value = scalar.value(stream.view());

  // Access device pointer
  int* d_ptr = scalar.data();

  // Pass to kernel
  trivial_kernel<<<1, 1, 0, stream.value()>>>(scalar.data());
  // [/device-scalar]
  // clang-format on

  stream.synchronize();
  assert(value == 42);
  (void)d_ptr;
}

void statistics_tracking()
{
  // clang-format off
  // [statistics-tracking]
  #include <rmm/mr/statistics_resource_adaptor.hpp>

  rmm::mr::cuda_async_memory_resource cuda_mr;
  rmm::mr::statistics_resource_adaptor stats_mr{cuda_mr};

  // Allocate using the statistics-wrapped resource
  rmm::cuda_stream stream;
  rmm::device_buffer buffer(1024, stream.view(), stats_mr);

  // Get statistics
  auto bytes = stats_mr.get_bytes_counter();
  std::cout << "Current bytes: " << bytes.value << "\n";
  std::cout << "Peak bytes: " << bytes.peak << "\n";
  std::cout << "Total bytes: " << bytes.total << "\n";
  // [/statistics-tracking]
  // clang-format on
}

void logging_example()
{
  // clang-format off
  // [logging]
  #include <rmm/mr/logging_resource_adaptor.hpp>

  rmm::mr::cuda_async_memory_resource cuda_mr;
  rmm::mr::logging_resource_adaptor log_mr{cuda_mr, "allocations.csv"};

  // Allocations through log_mr are logged to CSV
  rmm::cuda_stream stream;
  rmm::device_buffer buffer(1024, stream.view(), log_mr);
  // [/logging]
  // clang-format on

  assert(buffer.size() == 1024);
  std::remove("allocations.csv");
}

void composing_resources()
{
  // clang-format off
  // [composing-resources]
  #include <rmm/mr/cuda_memory_resource.hpp>
  #include <rmm/mr/pool_memory_resource.hpp>
  #include <rmm/mr/statistics_resource_adaptor.hpp>
  #include <rmm/mr/logging_resource_adaptor.hpp>

  // Base resource
  rmm::mr::cuda_memory_resource cuda_mr;

  // Add pool
  rmm::mr::pool_memory_resource pool_mr{cuda_mr, 1ULL << 30};

  // Add statistics
  rmm::mr::statistics_resource_adaptor stats_mr{pool_mr};

  // Add logging
  rmm::mr::logging_resource_adaptor log_mr{stats_mr, "log.csv"};

  // Use log_mr for allocations — all allocations are pooled, tracked, and logged
  rmm::cuda_stream stream;
  rmm::device_buffer buffer(1024, stream.view(), log_mr);
  // [/composing-resources]
  // clang-format on

  assert(buffer.size() == 1024);
  std::remove("log.csv");
}

void thrust_example()
{
  // clang-format off
  // [thrust]
  #include <rmm/exec_policy.hpp>
  #include <rmm/mr/cuda_async_memory_resource.hpp>
  #include <rmm/device_uvector.hpp>
  #include <thrust/sequence.h>
  #include <thrust/sort.h>

  rmm::mr::cuda_async_memory_resource mr;
  rmm::cuda_stream stream;
  rmm::device_uvector<int> vec(1000, stream.view(), mr);

  // Fill with descending values
  thrust::sequence(rmm::exec_policy_nosync(stream.view(), mr),
                   vec.begin(), vec.end(), vec.size() - 1, -1);

  // Sort — temporaries allocated from mr
  thrust::sort(rmm::exec_policy_nosync(stream.view(), mr), vec.begin(), vec.end());

  stream.synchronize();
  // [/thrust]
  // clang-format on
}

void multi_device_example()
{
  // clang-format off
  // [multi-device]
  #include <rmm/mr/cuda_async_memory_resource.hpp>
  #include <rmm/mr/per_device_resource.hpp>
  #include <rmm/cuda_device.hpp>
  #include <vector>

  int num_devices;
  cudaGetDeviceCount(&num_devices);

  // Store resources to maintain lifetime (resources are copyable value types)
  std::vector<rmm::mr::cuda_async_memory_resource> resources;

  for (int i = 0; i < num_devices; ++i) {
      // Set device BEFORE creating resource
      cudaSetDevice(i);

      // Create resource for this device
      resources.emplace_back();

      // Set as per-device resource ref
      rmm::mr::set_per_device_resource_ref(rmm::cuda_device_id{i}, resources.back());
  }

  // Use device 0
  cudaSetDevice(0);
  rmm::cuda_stream stream;
  rmm::device_buffer buffer(1024, stream.view());  // Uses device 0's resource
  // [/multi-device]
  // clang-format on

  assert(buffer.size() == 1024);
}

int main()
{
  explicit_resource();
  current_device_resource();
  device_buffer_example();
  device_uvector_example();
  device_scalar_example();
  statistics_tracking();
  logging_example();
  composing_resources();
  thrust_example();
  multi_device_example();

  std::cout << "All guide examples passed.\n";
  return 0;
}
