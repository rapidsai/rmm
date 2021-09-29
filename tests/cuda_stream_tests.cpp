/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

struct CudaStreamTest : public ::testing::Test {
};

TEST_F(CudaStreamTest, Equality)
{
  rmm::cuda_stream stream_a;
  auto const view_a       = stream_a.view();
  auto const view_default = rmm::cuda_stream_view{};

  EXPECT_EQ(stream_a, view_a);
  EXPECT_NE(stream_a, view_default);
  EXPECT_EQ(view_default, rmm::cuda_stream_view{});
  EXPECT_EQ(view_default, rmm::cuda_stream_default);
  EXPECT_NE(view_a, rmm::cuda_stream());
  EXPECT_NE(stream_a, rmm::cuda_stream());

  rmm::device_buffer buff{};
  EXPECT_EQ(buff.stream(), view_default);
}

TEST_F(CudaStreamTest, MoveConstructor)
{
  rmm::cuda_stream stream_a;
  auto const view_a         = stream_a.view();
  rmm::cuda_stream stream_b = std::move(stream_a);
  // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move)
  EXPECT_FALSE(stream_a.is_valid());  // Any other operations on stream_a are UB, may segfault
  EXPECT_EQ(stream_b, view_a);
}
