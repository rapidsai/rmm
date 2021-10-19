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

#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

using cuda_async_mr = rmm::mr::cuda_async_memory_resource;

TEST(PoolTest, ThrowIfNotSupported)
{
  auto construct_mr = []() { cuda_async_mr mr; };
#ifndef RMM_CUDA_MALLOC_ASYNC_SUPPORT
  EXPECT_THROW(construct_mr(), rmm::logic_error);
#else
  EXPECT_NO_THROW(construct_mr());
#endif
}

#if defined(RMM_CUDA_MALLOC_ASYNC_SUPPORT)
TEST(PoolTest, ExplicitInitialPoolSize)
{
  const auto pool_init_size{100};
  cuda_async_mr mr{pool_init_size};
  void* ptr = mr.allocate(pool_init_size);
  mr.deallocate(ptr, pool_init_size);
  RMM_CUDA_TRY(cudaDeviceSynchronize());
}

TEST(PoolTest, ExplicitReleaseThreshold)
{
  const auto pool_init_size{100};
  const auto pool_release_threshold{1000};
  cuda_async_mr mr{pool_init_size, pool_release_threshold};
  void* ptr = mr.allocate(pool_init_size);
  mr.deallocate(ptr, pool_init_size);
  RMM_CUDA_TRY(cudaDeviceSynchronize());
}

TEST(PoolTest, TakingOwnershipOfPool)
{
  cudaMemPoolProps poolProps = { };
  poolProps.allocType = cudaMemAllocationTypePinned;
  poolProps.location.id = rmm::detail::current_device().value();
  poolProps.location.type = cudaMemLocationTypeDevice;

  cudaMemPool_t memPool{};

  RMM_CUDA_TRY(cudaMemPoolCreate(&memPool, &poolProps));

  {

    const auto pool_init_size{100};
    cuda_async_mr mr{memPool};
    void* ptr = mr.allocate(pool_init_size);
    mr.deallocate(ptr, pool_init_size);
    RMM_CUDA_TRY(cudaDeviceSynchronize());

  }

  auto destroy_invalid_pool = [&](){
    auto result = cudaMemPoolDestroy(memPool);
    RMM_EXPECTS(result == cudaErrorInvalidValue, "Owning mr did not destroy owned pool");
  };

  EXPECT_NO_THROW(destroy_invalid_pool());
}

TEST(PoolTest, ThrowIfNullptrPool)
{
  auto construct_mr = []() {
    cudaMemPool_t memPool{nullptr};
    cuda_async_mr mr{memPool};
  };

  EXPECT_THROW(construct_mr(), rmm::logic_error);
}

TEST(PoolTest, ThrowIfDefaultPool)
{
  auto construct_mr = []() {
    cudaMemPool_t memPool{};
    RMM_CUDA_TRY(cudaDeviceGetDefaultMemPool(&memPool, rmm::detail::current_device().value()));

    cuda_async_mr mr{memPool};
  };

  EXPECT_THROW(construct_mr(), rmm::logic_error);
}


#endif

}  // namespace
}  // namespace rmm::test
