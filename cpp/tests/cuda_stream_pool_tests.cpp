/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <cstdint>

struct CudaStreamPoolTest : public ::testing::Test {
  rmm::cuda_stream_pool pool{};
};

TEST_F(CudaStreamPoolTest, ZeroSizePoolException)
{
  EXPECT_THROW(rmm::cuda_stream_pool pool{0}, rmm::logic_error);
}

TEST_F(CudaStreamPoolTest, Unequal)
{
  auto const stream_a = this->pool.get_stream();
  auto const stream_b = this->pool.get_stream();

  EXPECT_NE(stream_a, stream_b);
}

TEST_F(CudaStreamPoolTest, Nondefault)
{
  auto const stream_a = this->pool.get_stream();

  // pool streams are explicit, non-default streams
  EXPECT_FALSE(stream_a.is_default());
  EXPECT_FALSE(stream_a.is_per_thread_default());
}

TEST_F(CudaStreamPoolTest, ValidStreams)
{
  auto const stream_a = this->pool.get_stream();
  auto const stream_b = this->pool.get_stream();

  // Operations on the streams should work correctly and without throwing exceptions
  auto constexpr vector_size{100};
  auto vec1 = rmm::device_uvector<std::uint8_t>{vector_size, stream_a};
  RMM_CUDA_TRY(cudaMemsetAsync(vec1.data(), 0xcc, 100, stream_a.value()));
  stream_a.synchronize();

  auto vec2    = rmm::device_uvector<std::uint8_t>{vec1, stream_b};
  auto element = vec2.front_element(stream_b);
  EXPECT_EQ(element, 0xcc);
}

TEST_F(CudaStreamPoolTest, PoolSize) { EXPECT_GE(this->pool.get_pool_size(), 1); }

TEST_F(CudaStreamPoolTest, OutOfBoundLinearAccess)
{
  auto const stream_a = this->pool.get_stream(0);
  auto const stream_b = this->pool.get_stream(this->pool.get_pool_size());
  EXPECT_EQ(stream_a, stream_b);
}

TEST_F(CudaStreamPoolTest, ValidLinearAccess)
{
  auto const stream_a = this->pool.get_stream(0);
  auto const stream_b = this->pool.get_stream(1);
  EXPECT_NE(stream_a, stream_b);
}

TEST_F(CudaStreamPoolTest, CreateDefault)
{
  for (std::size_t i = 0; i < this->pool.get_pool_size(); i++) {
    auto stream = this->pool.get_stream(i);
    unsigned int flags;
    RMM_CUDA_TRY(cudaStreamGetFlags(stream.value(), &flags));
    EXPECT_EQ(flags, cudaStreamDefault);
  }
}

TEST_F(CudaStreamPoolTest, CreateNonBlocking)
{
  rmm::cuda_stream_pool pool{2, rmm::cuda_stream::flags::non_blocking};
  for (std::size_t i = 0; i < pool.get_pool_size(); i++) {
    auto stream = pool.get_stream(i);
    unsigned int flags;
    RMM_CUDA_TRY(cudaStreamGetFlags(stream.value(), &flags));
    EXPECT_EQ(flags, cudaStreamNonBlocking);
  }
}
