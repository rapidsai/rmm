/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Code examples for docs/user_guide/installation.md

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <iostream>

void test_installation()
{
  // clang-format off
  // [test-installation]
  #include <rmm/device_buffer.hpp>
  #include <rmm/mr/cuda_memory_resource.hpp>
  #include <rmm/mr/per_device_resource.hpp>
  #include <iostream>

  auto mr = rmm::mr::cuda_memory_resource{};
  rmm::mr::set_current_device_resource_ref(mr);

  rmm::device_buffer buf(100, rmm::cuda_stream_view{});
  std::cout << "Allocated " << buf.size() << " bytes\n";
  // [/test-installation]
  // clang-format on
}

int main()
{
  test_installation();
  return 0;
}
