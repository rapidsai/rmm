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

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

struct CudaStreamPoolTest : public ::testing::Test {
  rmm::cuda_stream_pool pool{};
};

TEST_F(CudaStreamPoolTest, Unequal)
{
  auto const stream_a = this->pool.get_stream();
  auto const stream_b = this->pool.get_stream();

  EXPECT_NE(stream_a, stream_b);
}

TEST_F(CudaStreamPoolTest, Nondefault)
{
  auto const stream_a = this->pool.get_stream();
  auto const stream_b = this->pool.get_stream();

  // pool streams are explicit, non-default streams
  EXPECT_FALSE(stream_a.is_default());
  EXPECT_FALSE(stream_a.is_per_thread_default());
}

TEST_F(CudaStreamPoolTest, ValidStreams)
{
  auto const stream_a = this->pool.get_stream();
  auto const stream_b = this->pool.get_stream();

  // Operations on the streams should work correctly and without throwing exceptions
  auto v = rmm::device_uvector<std::uint8_t>{100, stream_a};
  RMM_CUDA_TRY(cudaMemsetAsync(v.data(), 0xcc, 100, stream_a.value()));
  stream_a.synchronize();

  auto v2 = rmm::device_uvector<uint8_t>{v, stream_b};
  auto x  = v2.front_element(stream_b);
  EXPECT_EQ(x, 0xcc);
}
