/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Code examples for docs/user_guide/choosing_memory_resources.md
//
// Include directives that appear inside function bodies are intentional:
// they are no-ops (headers use #pragma once) and exist so that
// literalinclude snippets display the includes alongside the code.

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/prefetch_resource_adaptor.hpp>

#include <cassert>
#include <iostream>

void recommended_default()
{
  // clang-format off
  // [recommended-default]
  #include <rmm/mr/cuda_async_memory_resource.hpp>
  #include <rmm/device_buffer.hpp>

  rmm::mr::cuda_async_memory_resource mr;
  rmm::cuda_stream stream;
  rmm::device_buffer buffer(1024, stream.view(), mr);
  // [/recommended-default]
  // clang-format on

  assert(buffer.size() == 1024);
}

void managed_pool_prefetch()
{
  // clang-format off
  // [managed-pool-prefetch]
  #include <rmm/mr/managed_memory_resource.hpp>
  #include <rmm/mr/pool_memory_resource.hpp>
  #include <rmm/mr/prefetch_resource_adaptor.hpp>
  #include <rmm/aligned.hpp>
  #include <rmm/cuda_device.hpp>

  // Use 80% of GPU memory, rounded down to nearest 256 bytes
  auto [free_memory, total_memory] = rmm::available_device_memory();
  auto pool_size = rmm::align_down(static_cast<std::size_t>(total_memory * 0.8), 256);

  rmm::mr::managed_memory_resource managed_mr;
  rmm::mr::pool_memory_resource pool_mr{managed_mr, pool_size};
  rmm::mr::prefetch_resource_adaptor prefetch_mr{pool_mr};
  // [/managed-pool-prefetch]
  // clang-format on

  rmm::cuda_stream stream;
  rmm::device_buffer buffer(1024, stream.view(), prefetch_mr);
  assert(buffer.size() == 1024);
}

int main()
{
  recommended_default();
  managed_pool_prefetch();

  std::cout << "All choosing_memory_resources examples passed.\n";
  return 0;
}
