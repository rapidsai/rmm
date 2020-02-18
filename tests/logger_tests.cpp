/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>

#include <gtest/gtest.h>

TEST(First, first) {
  auto file_logger = spdlog::basic_logger_mt("rmm_logger", "logs/basic.txt");
  spdlog::set_default_logger(file_logger);
  spdlog::info("Welcome to spdlog version {}.{}.{}  !", SPDLOG_VER_MAJOR,
               SPDLOG_VER_MINOR, SPDLOG_VER_PATCH);
}

TEST(Adaptor, first) {
  rmm::mr::cuda_memory_resource upstream;

  rmm::mr::logging_resource_adaptor<rmm::mr::cuda_memory_resource> log_mr{
      &upstream, "logs/test.txt"};

  auto p = log_mr.allocate(100);
  log_mr.deallocate(p, 100);
}

TEST(Adaptor, factory) {
  rmm::mr::cuda_memory_resource upstream;

  auto log_mr = rmm::mr::make_logging_adaptor(&upstream, "logs/test.txt");

  auto p = log_mr.allocate(100);
  log_mr.deallocate(p, 100);
}