/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Code examples for docs/user_guide/logging.md
//
// Include directives that appear inside function bodies are intentional:
// they are no-ops (headers use #pragma once) and exist so that
// literalinclude snippets display the includes alongside the code.

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/logger.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/logging_resource_adaptor.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cassert>
#include <cstdio>
#include <iostream>

void logging_adaptor()
{
  // clang-format off
  // [logging-adaptor]
  #include <rmm/mr/cuda_async_memory_resource.hpp>
  #include <rmm/mr/logging_resource_adaptor.hpp>

  rmm::mr::cuda_async_memory_resource cuda_mr;
  rmm::mr::logging_resource_adaptor log_mr{cuda_mr, "memory_log.csv"};

  // Allocations through log_mr are logged to CSV
  rmm::cuda_stream stream;
  rmm::device_buffer buf1(1024, stream.view(), log_mr);
  rmm::device_buffer buf2(2048, stream.view(), log_mr);
  // [/logging-adaptor]
  // clang-format on

  std::remove("memory_log.csv");
}

void statistics_adaptor()
{
  // clang-format off
  // [statistics-adaptor]
  #include <rmm/mr/cuda_async_memory_resource.hpp>
  #include <rmm/mr/statistics_resource_adaptor.hpp>
  #include <iostream>

  rmm::mr::cuda_async_memory_resource cuda_mr;
  rmm::mr::statistics_resource_adaptor stats_mr{cuda_mr};

  // Allocate using the statistics-wrapped resource
  rmm::cuda_stream stream;
  rmm::device_buffer buf1(1024, stream.view(), stats_mr);
  rmm::device_buffer buf2(2048, stream.view(), stats_mr);

  // Get statistics
  auto bytes = stats_mr.get_bytes_counter();
  auto allocs = stats_mr.get_allocations_counter();
  std::cout << "Current bytes: " << bytes.value << "\n";
  std::cout << "Peak bytes: " << bytes.peak << "\n";
  std::cout << "Allocation count: " << allocs.value << "\n";
  // [/statistics-adaptor]
  // clang-format on
}

void debug_log_level()
{
  // clang-format off
  // [debug-log-level]
  #include <rmm/logger.hpp>

  rmm::default_logger().set_level(rapids_logger::level_enum::trace);
  // [/debug-log-level]
  // clang-format on

  // Reset to default
  rmm::default_logger().set_level(rapids_logger::level_enum::info);
}

void combining_features()
{
  // clang-format off
  // [combining-features]
  #include <rmm/mr/cuda_async_memory_resource.hpp>
  #include <rmm/mr/statistics_resource_adaptor.hpp>
  #include <rmm/mr/logging_resource_adaptor.hpp>
  #include <rmm/logger.hpp>

  // Set debug log level
  rmm::default_logger().set_level(rapids_logger::level_enum::debug);

  // Build resource stack: statistics + logging
  rmm::mr::cuda_async_memory_resource cuda_mr;
  rmm::mr::statistics_resource_adaptor stats_mr{cuda_mr};
  rmm::mr::logging_resource_adaptor log_mr{stats_mr, "events.csv"};

  // All allocations through log_mr are tracked and logged
  rmm::cuda_stream stream;
  rmm::device_buffer buffer(1024, stream.view(), log_mr);

  // Get statistics
  auto bytes = stats_mr.get_bytes_counter();
  std::cout << "Peak bytes: " << bytes.peak << "\n";
  // [/combining-features]
  // clang-format on

  // Reset to default
  rmm::default_logger().set_level(rapids_logger::level_enum::info);
  std::remove("events.csv");
}

int main()
{
  logging_adaptor();
  statistics_adaptor();
  debug_log_level();
  combining_features();

  std::cout << "All logging examples passed.\n";
  return 0;
}
