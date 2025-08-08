/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/prefetch_resource_adaptor.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <random>

using prefetch_adaptor = rmm::mr::prefetch_resource_adaptor<rmm::mr::device_memory_resource>;

template <typename MemoryResourceType>
struct PrefetchAdaptorTest : public ::testing::Test {
  rmm::cuda_stream stream{};
  std::size_t size{};
  MemoryResourceType mr{};

  PrefetchAdaptorTest()
  {
    std::default_random_engine generator;

    auto constexpr range_min{1000};
    auto constexpr range_max{100000};
    std::uniform_int_distribution<std::size_t> distribution(range_min, range_max);
    size = distribution(generator);
  }

  // Test that the memory range was last prefetched to the specified device
  void expect_prefetched(void const* ptr, std::size_t size, rmm::cuda_device_id device)
  {
    if constexpr (std::is_same_v<MemoryResourceType, rmm::mr::managed_memory_resource>) {
      // Skip the test if concurrent managed access is not supported
      if (!rmm::detail::concurrent_managed_access::is_supported()) {
        GTEST_SKIP() << "Skipping test: concurrent managed access not supported";
      }

      int prefetch_location{0};
      // See the CUDA documentation for cudaMemRangeGetAttribute
      // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g8048f6ea5ad77917444567656c140c5a
      // specifically for when cudaMemRangeAttribute::cudaMemRangeAttributeLastPrefetchLocation is
      // used.
      constexpr size_t prefetch_data_size = 4;
      RMM_CUDA_TRY(
        cudaMemRangeGetAttribute(&prefetch_location,
                                 prefetch_data_size,
                                 cudaMemRangeAttribute::cudaMemRangeAttributeLastPrefetchLocation,
                                 ptr,
                                 size));
      EXPECT_EQ(prefetch_location, device.value());
    }
  }
};

using resources = ::testing::Types<rmm::mr::cuda_memory_resource, rmm::mr::managed_memory_resource>;

TYPED_TEST_SUITE(PrefetchAdaptorTest, resources);

// The following tests simply test compilation and that there are no exceptions thrown
// due to prefetching non-managed memory.

TYPED_TEST(PrefetchAdaptorTest, PointerAndSize)
{
  auto* orig_device_resource = &this->mr;
  prefetch_adaptor prefetch_mr{orig_device_resource};
  rmm::device_buffer buff(this->size, this->stream, &prefetch_mr);
  // verify data range has been prefetched
  this->expect_prefetched(buff.data(), buff.size(), rmm::get_current_cuda_device());
  // verify that prefetching does not error
  rmm::prefetch(buff.data(), buff.size(), rmm::get_current_cuda_device(), this->stream);
  // reverify data range has been prefetched
  this->expect_prefetched(buff.data(), buff.size(), rmm::get_current_cuda_device());
}

TYPED_TEST(PrefetchAdaptorTest, NotPrefetchedWithoutAdaptor)
{
  // verify not prefetched without adaptor
  rmm::device_buffer buff(this->size, this->stream, &this->mr);
  this->expect_prefetched(buff.data(), buff.size(), rmm::cuda_device_id(cudaInvalidDeviceId));
}

TEST(PrefetchAdaptorTestNullUpstream, ThrowOnNullUpstream)
{
  auto construct_nullptr = []() { prefetch_adaptor mr{nullptr}; };
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);
}
