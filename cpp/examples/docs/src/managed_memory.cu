/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Code examples for docs/user_guide/managed_memory.md
//
// Include directives that appear inside function bodies are intentional:
// they are no-ops (headers use #pragma once) and exist so that
// literalinclude snippets display the includes alongside the code.

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/prefetch.hpp>

#include <cassert>
#include <iostream>

__global__ void trivial_kernel(void* data) {}

void prefetch_on_access()
{
  dim3 grid(1), block(1);

  // clang-format off
  // [prefetch-on-access]
  #include <rmm/mr/managed_memory_resource.hpp>
  #include <rmm/device_buffer.hpp>
  #include <rmm/prefetch.hpp>

  rmm::mr::managed_memory_resource managed_mr;
  rmm::cuda_stream stream;
  rmm::device_buffer buffer(1000000, stream.view(), managed_mr);

  // Prefetch to the current device on this stream
  rmm::prefetch(buffer.data(), buffer.size(),
                rmm::get_current_cuda_device(), stream.view());

  // Kernel on the same stream finds the data already resident
  trivial_kernel<<<grid, block, 0, stream.value()>>>(buffer.data());
  // [/prefetch-on-access]
  // clang-format on

  stream.synchronize();
}

int main()
{
  prefetch_on_access();

  std::cout << "All managed_memory examples passed.\n";
  return 0;
}
