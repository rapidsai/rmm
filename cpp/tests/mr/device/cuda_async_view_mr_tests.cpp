/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
#include <rmm/detail/cuda_memory_resource.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/cuda_async_view_memory_resource.hpp>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

using cuda_async_view_mr = rmm::mr::cuda_async_view_memory_resource;

// static property checks
static_assert(
  rmm::detail::polyfill::resource_with<cuda_async_view_mr, cuda::mr::device_accessible>);
static_assert(
  rmm::detail::polyfill::async_resource_with<cuda_async_view_mr, cuda::mr::device_accessible>);

TEST(PoolTest, UsePool)
{
  cudaMemPool_t memPool{};
  RMM_CUDA_TRY(cudaDeviceGetDefaultMemPool(&memPool, rmm::get_current_cuda_device().value()));

  const auto pool_init_size{100};
  cuda_async_view_mr mr{memPool};
  void* ptr = mr.allocate(pool_init_size);
  mr.deallocate(ptr, pool_init_size);
  RMM_CUDA_TRY(cudaDeviceSynchronize());
}

TEST(PoolTest, NotTakingOwnershipOfPool)
{
  cudaMemPoolProps poolProps = {};
  poolProps.allocType        = cudaMemAllocationTypePinned;
  poolProps.location.id      = rmm::get_current_cuda_device().value();
  poolProps.location.type    = cudaMemLocationTypeDevice;

  cudaMemPool_t memPool{};

  RMM_CUDA_TRY(cudaMemPoolCreate(&memPool, &poolProps));

  {
    const auto pool_init_size{100};
    cuda_async_view_mr mr{memPool};
    void* ptr = mr.allocate(pool_init_size);
    mr.deallocate(ptr, pool_init_size);
    RMM_CUDA_TRY(cudaDeviceSynchronize());
  }

  auto destroy_valid_pool = [&]() {
    auto result = cudaMemPoolDestroy(memPool);
    RMM_EXPECTS(result == cudaSuccess, "Pool wrapper did destroy pool");
  };

  EXPECT_NO_THROW(destroy_valid_pool());
}

TEST(PoolTest, ThrowIfNullptrPool)
{
  auto construct_mr = []() {
    cudaMemPool_t memPool{nullptr};
    cuda_async_view_mr mr{memPool};
  };

  EXPECT_THROW(construct_mr(), rmm::logic_error);
}

}  // namespace
}  // namespace rmm::test
