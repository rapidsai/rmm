/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

class HWDecompressTest : public ::testing::Test {
 protected:
  static void check_decompress_capable(void* ptr)
  {
    int driver_version{};
    RMM_CUDA_TRY(cudaDriverGetVersion(&driver_version));
    auto const min_hw_decompress_version{12080};
    if (driver_version >= min_hw_decompress_version) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
      bool is_capable{};
      auto err =
        cuPointerGetAttribute(static_cast<void*>(&is_capable),
                              CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE,
                              // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                              reinterpret_cast<CUdeviceptr>(ptr));
      EXPECT_EQ(err, CUDA_SUCCESS);
      EXPECT_TRUE(is_capable);
#endif
    }
  }
};

TEST_F(HWDecompressTest, CudaMalloc)
{
  const auto allocation_size{100};
  rmm::mr::cuda_memory_resource mr{};
  void* ptr = mr.allocate(allocation_size);
  HWDecompressTest::check_decompress_capable(ptr);
  mr.deallocate(ptr, allocation_size);
  RMM_CUDA_TRY(cudaDeviceSynchronize());
}

TEST_F(HWDecompressTest, CudaMallocAsync)
{
  if (!rmm::detail::runtime_async_alloc::is_supported()) {
    GTEST_SKIP() << "Skipping since cudaMallocAsync not supported with this CUDA "
                 << "driver/runtime version";
  }
  const auto pool_init_size{100};
  rmm::mr::cuda_async_memory_resource mr{pool_init_size};
  void* ptr = mr.allocate(pool_init_size);
  HWDecompressTest::check_decompress_capable(ptr);
  mr.deallocate(ptr, pool_init_size);
  RMM_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace
}  // namespace rmm::test
