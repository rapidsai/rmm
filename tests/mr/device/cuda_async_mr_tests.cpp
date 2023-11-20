/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

using cuda_async_mr = rmm::mr::cuda_async_memory_resource;
static_assert(cuda::mr::resource_with<cuda_async_mr, cuda::mr::device_accessible>);
static_assert(cuda::mr::async_resource_with<cuda_async_mr, cuda::mr::device_accessible>);

class AsyncMRTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    if (!rmm::detail::async_alloc::is_supported()) {
      GTEST_SKIP() << "Skipping tests since cudaMallocAsync not supported with this CUDA "
                   << "driver/runtime version";
    }
  }
};

TEST_F(AsyncMRTest, ThrowIfNotSupported)
{
  auto construct_mr = []() { cuda_async_mr mr; };
#ifndef RMM_CUDA_MALLOC_ASYNC_SUPPORT
  EXPECT_THROW(construct_mr(), rmm::logic_error);
#else
  EXPECT_NO_THROW(construct_mr());
#endif
}

#if defined(RMM_CUDA_MALLOC_ASYNC_SUPPORT)
TEST_F(AsyncMRTest, ExplicitInitialPoolSize)
{
  const auto pool_init_size{100};
  cuda_async_mr mr{pool_init_size};
  void* ptr = mr.allocate(pool_init_size);
  mr.deallocate(ptr, pool_init_size);
  RMM_CUDA_TRY(cudaDeviceSynchronize());
}

TEST_F(AsyncMRTest, ExplicitReleaseThreshold)
{
  const auto pool_init_size{100};
  const auto pool_release_threshold{1000};
  cuda_async_mr mr{pool_init_size, pool_release_threshold};
  void* ptr = mr.allocate(pool_init_size);
  mr.deallocate(ptr, pool_init_size);
  RMM_CUDA_TRY(cudaDeviceSynchronize());
}

TEST_F(AsyncMRTest, DifferentPoolsUnequal)
{
  const auto pool_init_size{100};
  const auto pool_release_threshold{1000};
  cuda_async_mr mr1{pool_init_size, pool_release_threshold};
  cuda_async_mr mr2{pool_init_size, pool_release_threshold};
  EXPECT_FALSE(mr1.is_equal(mr2));
}

#endif

}  // namespace
}  // namespace rmm::test
