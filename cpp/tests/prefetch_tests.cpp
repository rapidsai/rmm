/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "driver_types.h"
#include "rmm/cuda_device.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/prefetch.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <random>

template <typename MemoryResourceType>
struct PrefetchTest : public ::testing::Test {
  rmm::cuda_stream stream{};
  std::size_t size{};
  MemoryResourceType mr{};

  PrefetchTest()
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
    if (!rmm::detail::concurrent_managed_access::is_supported()) {
      GTEST_SKIP() << "Skipping test: concurrent managed access not supported";
    }

    // See the CUDA documentation for cudaMemRangeGetAttribute
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g8048f6ea5ad77917444567656c140c5a
    // specifically for when cudaMemRangeAttribute::cudaMemRangeAttributeLastPrefetchLocation is
    // used.
    if constexpr (std::is_same_v<MemoryResourceType, rmm::mr::managed_memory_resource>) {
      constexpr size_t prefetch_data_size = 4;
      int prefetch_location{0};
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

TYPED_TEST_SUITE(PrefetchTest, resources);

// The following tests simply test compilation and that there are no exceptions thrown
// due to prefetching non-managed memory.

TYPED_TEST(PrefetchTest, PointerAndSize)
{
  rmm::device_buffer buff(this->size, this->stream, &this->mr);
  // verify not prefetched before prefetching
  this->expect_prefetched(buff.data(), buff.size(), rmm::cuda_device_id(cudaInvalidDeviceId));
  rmm::prefetch(buff.data(), buff.size(), rmm::get_current_cuda_device(), this->stream);
  // verify data range has been prefetched
  this->expect_prefetched(buff.data(), buff.size(), rmm::get_current_cuda_device());
}

TYPED_TEST(PrefetchTest, DeviceUVector)
{
  {
    rmm::device_uvector<int> uvec(this->size, this->stream, &this->mr);
    rmm::prefetch<int>(uvec, rmm::get_current_cuda_device(), this->stream);
    this->expect_prefetched(uvec.data(), uvec.size() * sizeof(int), rmm::get_current_cuda_device());
  }

  // test iterator range of part of the vector (implicitly constructs a span)
  {
    rmm::device_uvector<int> uvec(this->size, this->stream, &this->mr);
    rmm::prefetch<int>({uvec.begin(), std::next(uvec.begin(), this->size / 2)},  // span
                       rmm::get_current_cuda_device(),
                       this->stream);
    this->expect_prefetched(
      uvec.data(), this->size / 2 * sizeof(int), rmm::get_current_cuda_device());
  }
}

TYPED_TEST(PrefetchTest, DeviceScalar)
{
  rmm::device_scalar<int> scalar(this->stream, &this->mr);
  // Implicitly constructs a span from the data and size
  rmm::prefetch<int>({scalar.data(), scalar.size()}, rmm::get_current_cuda_device(), this->stream);
  this->expect_prefetched(scalar.data(), sizeof(int), rmm::get_current_cuda_device());
}
