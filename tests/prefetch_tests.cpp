/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
};

using resources = ::testing::Types<rmm::mr::cuda_memory_resource, rmm::mr::managed_memory_resource>;

TYPED_TEST_CASE(PrefetchTest, resources);

// The following tests simply test compilation and that there are no exceptions thrown
// due to prefetching non-managed memory.

TYPED_TEST(PrefetchTest, PointerAndSize)
{
  rmm::device_buffer buff(this->size, this->stream, &this->mr);
  rmm::prefetch(buff.data(), buff.size(), rmm::get_current_cuda_device(), this->stream);
}

TYPED_TEST(PrefetchTest, DeviceBuffer)
{
  rmm::device_buffer buff(this->size, this->stream, &this->mr);
  rmm::prefetch<char>(buff, rmm::get_current_cuda_device(), this->stream);
  rmm::prefetch<char const>(buff, rmm::get_current_cuda_device(), this->stream);  // const version
}

TYPED_TEST(PrefetchTest, DeviceUVector)
{
  rmm::device_uvector<int> uvec(this->size, this->stream, &this->mr);
  rmm::prefetch<int>(uvec, rmm::get_current_cuda_device(), this->stream);

  // test iterator range of part of the vector (implicitly constructs a span)
  rmm::prefetch<int>({uvec.begin(), std::next(uvec.begin(), this->size / 2)},
                     rmm::get_current_cuda_device(),
                     this->stream);
}

TYPED_TEST(PrefetchTest, DeviceScalar)
{
  rmm::device_scalar<int> scalar(this->stream, &this->mr);
  rmm::prefetch<int>(scalar, rmm::get_current_cuda_device(), this->stream);
}
