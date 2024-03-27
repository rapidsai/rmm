/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "../../byte_literals.hpp"
#include "test_utils.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
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
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <utility>

using resource_ref = cuda::mr::resource_ref<cuda::mr::device_accessible>;

namespace rmm::test {

enum size_in_bytes : size_t {};

constexpr auto default_num_allocations{100};
constexpr size_in_bytes default_max_size{5_MiB};

struct allocation {
  void* ptr{nullptr};
  std::size_t size{0};
  allocation(void* ptr, std::size_t size) : ptr{ptr}, size{size} {}
  allocation() = default;
};

// Various test functions, shared between single-threaded and multithreaded tests.

inline void test_get_current_device_resource()
{
  EXPECT_NE(nullptr, rmm::mr::get_current_device_resource());
  void* ptr = rmm::mr::get_current_device_resource()->allocate(1_MiB);
  EXPECT_NE(nullptr, ptr);
  EXPECT_TRUE(is_properly_aligned(ptr));
  EXPECT_TRUE(is_device_accessible_memory(ptr));
  rmm::mr::get_current_device_resource()->deallocate(ptr, 1_MiB);
}

inline void test_allocate(resource_ref ref, std::size_t bytes)
{
  try {
    void* ptr = ref.allocate(bytes);
    EXPECT_NE(nullptr, ptr);
    EXPECT_TRUE(is_properly_aligned(ptr));
    EXPECT_TRUE(is_device_accessible_memory(ptr));
    ref.deallocate(ptr, bytes);
  } catch (rmm::out_of_memory const& e) {
    EXPECT_NE(std::string{e.what()}.find("out_of_memory"), std::string::npos);
  }
}

inline void test_allocate_async(rmm::device_async_resource_ref ref,
                                std::size_t bytes,
                                cuda_stream_view stream = {})
{
  try {
    void* ptr = ref.allocate_async(bytes, stream);
    if (not stream.is_default()) { stream.synchronize(); }
    EXPECT_NE(nullptr, ptr);
    EXPECT_TRUE(is_properly_aligned(ptr));
    EXPECT_TRUE(is_device_accessible_memory(ptr));
    ref.deallocate_async(ptr, bytes, stream);
    if (not stream.is_default()) { stream.synchronize(); }
  } catch (rmm::out_of_memory const& e) {
    EXPECT_NE(std::string{e.what()}.find("out_of_memory"), std::string::npos);
  }
}

// Simple reproducer for https://github.com/rapidsai/rmm/issues/861
inline void concurrent_allocations_are_different(resource_ref ref)
{
  const auto size{8_B};
  void* ptr1 = ref.allocate(size);
  void* ptr2 = ref.allocate(size);

  EXPECT_NE(ptr1, ptr2);

  ref.deallocate(ptr1, size);
  ref.deallocate(ptr2, size);
}

inline void concurrent_async_allocations_are_different(rmm::device_async_resource_ref ref,
                                                       cuda_stream_view stream)
{
  const auto size{8_B};
  void* ptr1 = ref.allocate_async(size, stream);
  void* ptr2 = ref.allocate_async(size, stream);

  EXPECT_NE(ptr1, ptr2);

  ref.deallocate_async(ptr1, size, stream);
  ref.deallocate_async(ptr2, size, stream);
}

inline void test_various_allocations(resource_ref ref)
{
  // test allocating zero bytes on non-default stream
  {
    void* ptr = ref.allocate(0);
    EXPECT_NO_THROW(ref.deallocate(ptr, 0));
  }

  test_allocate(ref, 4_B);
  test_allocate(ref, 1_KiB);
  test_allocate(ref, 1_MiB);
  test_allocate(ref, 1_GiB);

  // should fail to allocate too much
  {
    void* ptr{nullptr};
    EXPECT_THROW(ptr = ref.allocate(1_PiB), rmm::out_of_memory);
    EXPECT_EQ(nullptr, ptr);

    // test e.what();
    try {
      ptr = ref.allocate(1_PiB);
    } catch (rmm::out_of_memory const& e) {
      EXPECT_NE(std::string{e.what()}.find("out_of_memory"), std::string::npos);
    }
  }
}

inline void test_various_async_allocations(rmm::device_async_resource_ref ref,
                                           cuda_stream_view stream)
{
  // test allocating zero bytes on non-default stream
  {
    void* ptr = ref.allocate_async(0, stream);
    stream.synchronize();
    EXPECT_NO_THROW(ref.deallocate_async(ptr, 0, stream));
    stream.synchronize();
  }

  test_allocate_async(ref, 4_B, stream);
  test_allocate_async(ref, 1_KiB, stream);
  test_allocate_async(ref, 1_MiB, stream);
  test_allocate_async(ref, 1_GiB, stream);

  // should fail to allocate too much
  {
    void* ptr{nullptr};
    EXPECT_THROW(ptr = ref.allocate_async(1_PiB, stream), rmm::out_of_memory);
    EXPECT_EQ(nullptr, ptr);

    // test e.what();
    try {
      ptr = ref.allocate_async(1_PiB, stream);
    } catch (rmm::out_of_memory const& e) {
      EXPECT_NE(std::string{e.what()}.find("out_of_memory"), std::string::npos);
    }
  }
}

inline void test_random_allocations(resource_ref ref,
                                    std::size_t num_allocations = default_num_allocations,
                                    size_in_bytes max_size      = default_max_size)
{
  std::vector<allocation> allocations(num_allocations);

  std::default_random_engine generator;
  std::uniform_int_distribution<std::size_t> distribution(1, max_size);

  // num_allocations allocations from [0,max_size)
  std::for_each(
    allocations.begin(), allocations.end(), [&generator, &distribution, &ref](allocation& alloc) {
      alloc.size = distribution(generator);
      EXPECT_NO_THROW(alloc.ptr = ref.allocate(alloc.size));
      EXPECT_NE(nullptr, alloc.ptr);
      EXPECT_TRUE(is_properly_aligned(alloc.ptr));
    });

  std::for_each(allocations.begin(), allocations.end(), [&ref](allocation& alloc) {
    EXPECT_NO_THROW(ref.deallocate(alloc.ptr, alloc.size));
  });
}

inline void test_random_async_allocations(rmm::device_async_resource_ref ref,
                                          std::size_t num_allocations = default_num_allocations,
                                          size_in_bytes max_size      = default_max_size,
                                          cuda_stream_view stream     = {})
{
  std::vector<allocation> allocations(num_allocations);

  std::default_random_engine generator;
  std::uniform_int_distribution<std::size_t> distribution(1, max_size);

  // num_allocations allocations from [0,max_size)
  std::for_each(allocations.begin(),
                allocations.end(),
                [&generator, &distribution, &ref, stream](allocation& alloc) {
                  alloc.size = distribution(generator);
                  EXPECT_NO_THROW(alloc.ptr = ref.allocate(alloc.size));
                  if (not stream.is_default()) { stream.synchronize(); }
                  EXPECT_NE(nullptr, alloc.ptr);
                  EXPECT_TRUE(is_properly_aligned(alloc.ptr));
                });

  std::for_each(allocations.begin(), allocations.end(), [stream, &ref](allocation& alloc) {
    EXPECT_NO_THROW(ref.deallocate(alloc.ptr, alloc.size));
    if (not stream.is_default()) { stream.synchronize(); }
  });
}

inline void test_mixed_random_allocation_free(resource_ref ref,
                                              size_in_bytes max_size = default_max_size)
{
  std::default_random_engine generator;
  constexpr std::size_t num_allocations{100};

  std::uniform_int_distribution<std::size_t> size_distribution(1, max_size);

  constexpr int allocation_probability{53};  // percent
  constexpr int max_probability{99};
  std::uniform_int_distribution<int> op_distribution(0, max_probability);
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
      std::size_t size = size_distribution(generator);
      active_allocations++;
      allocation_count++;
      EXPECT_NO_THROW(allocations.emplace_back(ref.allocate(size), size));
      auto new_allocation = allocations.back();
      EXPECT_NE(nullptr, new_allocation.ptr);
      EXPECT_TRUE(is_properly_aligned(new_allocation.ptr));
    } else {
      auto const index = static_cast<int>(index_distribution(generator) % active_allocations);
      active_allocations--;
      allocation to_free = allocations[index];
      allocations.erase(std::next(allocations.begin(), index));
      EXPECT_NO_THROW(ref.deallocate(to_free.ptr, to_free.size));
    }
  }

  EXPECT_EQ(active_allocations, 0);
  EXPECT_EQ(allocations.size(), active_allocations);
}

inline void test_mixed_random_async_allocation_free(rmm::device_async_resource_ref ref,
                                                    size_in_bytes max_size  = default_max_size,
                                                    cuda_stream_view stream = {})
{
  std::default_random_engine generator;
  constexpr std::size_t num_allocations{100};

  std::uniform_int_distribution<std::size_t> size_distribution(1, max_size);

  constexpr int allocation_probability{53};  // percent
  constexpr int max_probability{99};
  std::uniform_int_distribution<int> op_distribution(0, max_probability);
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
      std::size_t size = size_distribution(generator);
      active_allocations++;
      allocation_count++;
      EXPECT_NO_THROW(allocations.emplace_back(ref.allocate_async(size, stream), size));
      auto new_allocation = allocations.back();
      EXPECT_NE(nullptr, new_allocation.ptr);
      EXPECT_TRUE(is_properly_aligned(new_allocation.ptr));
    } else {
      auto const index = static_cast<int>(index_distribution(generator) % active_allocations);
      active_allocations--;
      allocation to_free = allocations[index];
      allocations.erase(std::next(allocations.begin(), index));
      EXPECT_NO_THROW(ref.deallocate_async(to_free.ptr, to_free.size, stream));
    }
  }

  EXPECT_EQ(active_allocations, 0);
  EXPECT_EQ(allocations.size(), active_allocations);
}

using MRFactoryFunc = std::function<std::shared_ptr<rmm::mr::device_memory_resource>()>;

/// Encapsulates a `device_memory_resource` factory function and associated name
struct mr_factory {
  mr_factory(std::string name, MRFactoryFunc factory)
    : name{std::move(name)}, factory{std::move(factory)}
  {
  }

  std::string name;       ///< Name to associate with tests that use this factory
  MRFactoryFunc factory;  ///< Factory function that returns shared_ptr to `device_memory_resource`
                          ///< instance to use in test
};

/// Test fixture class value-parameterized on different `mr_factory`s
struct mr_ref_test : public ::testing::TestWithParam<mr_factory> {
  void SetUp() override
  {
    auto factory = GetParam().factory;
    mr           = factory();
    if (mr == nullptr) {
      GTEST_SKIP() << "Skipping tests since the memory resource is not supported with this CUDA "
                   << "driver/runtime version";
    }
    ref = rmm::device_async_resource_ref{*mr};
  }

  std::shared_ptr<rmm::mr::device_memory_resource> mr;  ///< Pointer to resource to use in tests
  rmm::device_async_resource_ref ref{*mr};
  rmm::cuda_stream stream{};
};

struct mr_ref_allocation_test : public mr_ref_test {};

/// MR factory functions
inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

inline auto make_host_pinned() { return std::make_shared<rmm::mr::pinned_host_memory_resource>(); }

inline auto make_cuda_async()
{
  if (rmm::detail::async_alloc::is_supported()) {
    return std::make_shared<rmm::mr::cuda_async_memory_resource>();
  }
  return std::shared_ptr<rmm::mr::cuda_async_memory_resource>{nullptr};
}

inline auto make_managed() { return std::make_shared<rmm::mr::managed_memory_resource>(); }

inline auto make_pool()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
    make_cuda(), rmm::percent_of_free_device_memory(50));
}

inline auto make_host_pinned_pool()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
    make_host_pinned(), 2_GiB, 8_GiB);
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
  auto const bin_range_start{18};
  auto const bin_range_end{22};

  auto mr = rmm::mr::make_owning_wrapper<rmm::mr::binning_memory_resource>(
    pool, bin_range_start, bin_range_end);
  return mr;
}

}  // namespace rmm::test
