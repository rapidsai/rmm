/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "../../byte_literals.hpp"
#include "rmm/cuda_stream_view.hpp"

#include <benchmarks/utilities/log_parser.hpp>

#include <rmm/detail/error.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <gtest/gtest.h>

#include <sys/stat.h>

namespace rmm::test {

namespace {
using cuda_mr  = rmm::mr::cuda_memory_resource;
using arena_mr = rmm::mr::arena_memory_resource<rmm::mr::cuda_memory_resource>;

TEST(ArenaTest, ThrowOnNullUpstream)
{
  auto construct_nullptr = []() { arena_mr mr{nullptr}; };
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);
}

TEST(ArenaTest, ThrowMaxLessThanInitial)
{
  // Make sure first argument is enough larger than the second that alignment rounding doesn't
  // make them equal
  auto max_less_than_initial = []() {
    const auto initial{4_MiB};
    const auto maximum{2_MiB};
    cuda_mr cuda;
    arena_mr mr{&cuda, initial, maximum};
  };
  EXPECT_THROW(max_less_than_initial(), rmm::logic_error);
}

TEST(ArenaTest, DumpLogOnFailure)
{
  cuda_mr cuda;
  arena_mr mr{&cuda, 1_MiB, 4_MiB, true};

  {  // make the log interesting
    std::vector<std::thread> threads;
    std::size_t num_threads{4};
    threads.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
      threads.emplace_back(std::thread([&] {
        void* ptr = mr.allocate(32_KiB);
        mr.deallocate(ptr, 32_KiB);
      }));
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }

  EXPECT_THROW(mr.allocate(8_MiB), rmm::out_of_memory);

  struct stat file_status {
  };
  EXPECT_EQ(stat("rmm_arena_memory_dump.log", &file_status), 0);
  EXPECT_GE(file_status.st_size, 0);
}

TEST(ArenaTest, FeatureSupport)
{
  cuda_mr cuda;
  arena_mr mr{&cuda, 1_MiB, 4_MiB};
  EXPECT_TRUE(mr.supports_streams());
  EXPECT_FALSE(mr.supports_get_mem_info());
  auto [free, total] = mr.get_mem_info(rmm::cuda_stream_default);
  EXPECT_EQ(free, 0);
  EXPECT_EQ(total, 0);
}

}  // namespace
}  // namespace rmm::test
