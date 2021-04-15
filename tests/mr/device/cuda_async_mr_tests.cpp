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

#include <rmm/detail/error.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>

#include <gtest/gtest.h>

namespace rmm {
namespace test {
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
TEST(PoolTest, ExplicitInitialPoolSize){
    cuda_async_mr mr{100};
    void* p;
    EXPECT_NO_THROW(p = mr.allocate(100));
    EXPECT_NO_THROW(mr.deallocate(p,100));
    RMM_CUDA_TRY(cudaDeviceSynchronize());
}

TEST(PoolTest, ExplicitReleaseThreshold){
    cuda_async_mr mr{100, 1000};
    void* p;
    EXPECT_NO_THROW(p = mr.allocate(100));
    EXPECT_NO_THROW(mr.deallocate(p,100));
    RMM_CUDA_TRY(cudaDeviceSynchronize());
}


#endif

}  // namespace
}  // namespace test
}  // namespace rmm
