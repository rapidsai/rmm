/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Code examples for docs/user_guide/introduction.md

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cassert>
#include <iostream>

void basic_example()
{
  // clang-format off
  // [basic-example]
  rmm::mr::cuda_async_memory_resource mr;
  rmm::cuda_stream stream;
  rmm::device_buffer buffer(1024, stream.view(), mr);
  // [/basic-example]
  // clang-format on

  assert(buffer.size() == 1024);
}

int main()
{
  basic_example();

  std::cout << "All introduction examples passed.\n";
  return 0;
}
