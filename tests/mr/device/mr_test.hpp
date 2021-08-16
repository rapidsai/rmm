/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#pragma once

#include <gtest/gtest.h>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/fixed_size_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>

namespace rmm {
namespace test {

inline bool is_pointer_aligned(void* p, std::size_t alignment = 256)
{
  return (0 == reinterpret_cast<uintptr_t>(p) % alignment);
}

/**
 * @brief Returns if a pointer points to a device memory or managed memory
 * allocation.
 */
inline bool is_device_memory(void* p)
{
  cudaPointerAttributes attributes{};
  if (cudaSuccess != cudaPointerGetAttributes(&attributes, p)) { return false; }
#if CUDART_VERSION < 10000  // memoryType is deprecated in CUDA 10
  return attributes.memoryType == cudaMemoryTypeDevice;
#else
  return (attributes.type == cudaMemoryTypeDevice) or (attributes.type == cudaMemoryTypeManaged);
#endif
}

// some useful allocation sizes
constexpr long operator""_B(unsigned long long const x) { return x; }
constexpr long operator""_KiB(unsigned long long const x) { return x * (long{1} << 10); }
constexpr long operator""_MiB(unsigned long long const x) { return x * (long{1} << 20); }
constexpr long operator""_GiB(unsigned long long const x) { return x * (long{1} << 30); }
constexpr long operator""_TiB(unsigned long long const x) { return x * (long{1} << 40); }
constexpr long operator""_PiB(unsigned long long const x) { return x * (long{1} << 50); }

struct allocation {
  void* p{nullptr};
  std::size_t size{0};
  allocation(void* _p, std::size_t _size) : p{_p}, size{_size} {}
  allocation() = default;
};

// Various test functions, shared between single-threaded and multithreaded tests.

inline void test_get_current_device_resource()
{
  EXPECT_NE(nullptr, rmm::mr::get_current_device_resource());
  void* p{nullptr};
  EXPECT_NO_THROW(p = rmm::mr::get_current_device_resource()->allocate(1_MiB));
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_pointer_aligned(p));
  EXPECT_TRUE(is_device_memory(p));
  EXPECT_NO_THROW(rmm::mr::get_current_device_resource()->deallocate(p, 1_MiB));
}

inline void test_allocate(rmm::mr::device_memory_resource* mr,
                          std::size_t bytes,
                          cuda_stream_view stream = {})
{
  void* p{nullptr};
  EXPECT_NO_THROW(p = mr->allocate(bytes));
  if (not stream.is_default()) stream.synchronize();
  EXPECT_NE(nullptr, p);
  EXPECT_TRUE(is_pointer_aligned(p));
  EXPECT_TRUE(is_device_memory(p));
  EXPECT_NO_THROW(mr->deallocate(p, bytes));
  if (not stream.is_default()) stream.synchronize();
}

inline void test_various_allocations(rmm::mr::device_memory_resource* mr, cuda_stream_view stream)
{
  // test allocating zero bytes on non-default stream
  {
    void* p{nullptr};
    EXPECT_NO_THROW(p = mr->allocate_async(0, stream));
    stream.synchronize();
    EXPECT_NO_THROW(mr->deallocate_async(p, 0, stream));
    stream.synchronize();
  }

  test_allocate(mr, 4_B, stream);
  test_allocate(mr, 1_KiB, stream);
  test_allocate(mr, 1_MiB, stream);
  test_allocate(mr, 1_GiB, stream);

  // should fail to allocate too much
  {
    void* p{nullptr};
    EXPECT_THROW(p = mr->allocate_async(1_PiB, stream), rmm::bad_alloc);
    EXPECT_EQ(nullptr, p);
  }
}

inline void test_random_allocations(rmm::mr::device_memory_resource* mr,
                                    std::size_t num_allocations = 100,
                                    std::size_t max_size        = 5_MiB,
                                    cuda_stream_view stream     = {})
{
  std::vector<allocation> allocations(num_allocations);

  std::default_random_engine generator;
  std::uniform_int_distribution<std::size_t> distribution(1, max_size);

  // 100 allocations from [0,5MB)
  std::for_each(
    allocations.begin(), allocations.end(), [&generator, &distribution, stream, mr](allocation& a) {
      a.size = distribution(generator);
      EXPECT_NO_THROW(a.p = mr->allocate_async(a.size, stream));
      if (not stream.is_default()) stream.synchronize();
      EXPECT_NE(nullptr, a.p);
      EXPECT_TRUE(is_pointer_aligned(a.p));
    });

  std::for_each(allocations.begin(), allocations.end(), [stream, mr](allocation& a) {
    EXPECT_NO_THROW(mr->deallocate_async(a.p, a.size, stream));
    if (not stream.is_default()) stream.synchronize();
  });
}

inline void test_mixed_random_allocation_free(rmm::mr::device_memory_resource* mr,
                                              std::size_t max_size    = 5_MiB,
                                              cuda_stream_view stream = {})
{
  std::default_random_engine generator;
  constexpr std::size_t num_allocations{100};

  std::uniform_int_distribution<std::size_t> size_distribution(1, max_size);

  constexpr int allocation_probability = 53;  // percent
  std::uniform_int_distribution<int> op_distribution(0, 99);
  std::uniform_int_distribution<int> index_distribution(0, num_allocations - 1);

  std::size_t active_allocations{0};
  std::size_t allocation_count{0};

  std::vector<allocation> allocations;

  for (std::size_t i = 0; i < num_allocations * 2; ++i) {
    bool do_alloc = true;
    if (active_allocations > 0) {
      int chance = op_distribution(generator);
      do_alloc   = (chance < allocation_probability) && (allocation_count < num_allocations);
    }

    if (do_alloc) {
      size_t size = size_distribution(generator);
      active_allocations++;
      allocation_count++;
      EXPECT_NO_THROW(allocations.emplace_back(mr->allocate_async(size, stream), size));
      auto new_allocation = allocations.back();
      EXPECT_NE(nullptr, new_allocation.p);
      EXPECT_TRUE(is_pointer_aligned(new_allocation.p));
    } else {
      size_t index = index_distribution(generator) % active_allocations;
      active_allocations--;
      allocation to_free = allocations[index];
      allocations.erase(std::next(allocations.begin(), index));
      EXPECT_NO_THROW(mr->deallocate_async(to_free.p, to_free.size, stream));
    }
  }

  EXPECT_EQ(active_allocations, 0);
  EXPECT_EQ(allocations.size(), active_allocations);
}

using MRFactoryFunc = std::function<std::shared_ptr<rmm::mr::device_memory_resource>()>;

/// Encapsulates a `device_memory_resource` factory function and associated name
struct mr_factory {
  mr_factory(std::string const& name, MRFactoryFunc f) : name{name}, f{f} {}

  std::string name;  ///< Name to associate with tests that use this factory
  MRFactoryFunc f;   ///< Factory function that returns shared_ptr to `device_memory_resource`
                     ///< instance to use in test
};

/// Test fixture class value-parameterized on different `mr_factory`s
struct mr_test : public ::testing::TestWithParam<mr_factory> {
  void SetUp() override
  {
    auto factory = GetParam().f;
    mr           = factory();
  }

  std::shared_ptr<rmm::mr::device_memory_resource> mr;  ///< Pointer to resource to use in tests
  rmm::cuda_stream stream{};
};

/// MR factory functions
inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

inline auto make_cuda_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

inline auto make_managed() { return std::make_shared<rmm::mr::managed_memory_resource>(); }

inline auto make_pool()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_cuda());
}

inline auto make_arena()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::arena_memory_resource>(make_cuda());
}

inline auto make_fixed_size()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::fixed_size_memory_resource>(make_cuda());
}

inline auto make_binning()
{
  auto pool = make_pool();
  // Add a binning_memory_resource with fixed-size bins of sizes 256, 512, 1024, 2048 and 4096KiB
  // Larger allocations will use the pool resource
  auto mr = rmm::mr::make_owning_wrapper<rmm::mr::binning_memory_resource>(pool, 18, 22);
  return mr;
}

}  // namespace test
}  // namespace rmm
