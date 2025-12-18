/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Tests for resource_ref conversion when used with custom allocators
// This tests patterns similar to cudf's device_scalar bounce buffer

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/host_vector.h>

#include <gtest/gtest.h>

namespace rmm::test {
namespace {

// Host allocator that takes a resource_ref (similar to cudf's rmm_host_allocator)
template <typename T>
class host_allocator {
 public:
  using value_type = T;

  host_allocator() = delete;

  template <class... Properties>
  host_allocator(cuda::mr::resource_ref<cuda::mr::host_accessible, Properties...> mr,
                 rmm::cuda_stream_view stream)
    : mr_(mr), stream_(stream)
  {
  }

  host_allocator(host_allocator const& other) = default;
  host_allocator(host_allocator&& other)      = default;

  T* allocate(std::size_t n)
  {
    auto const result = mr_.allocate(stream_, n * sizeof(T));
    stream_.synchronize();
    return static_cast<T*>(result);
  }

  void deallocate(T* p, std::size_t n) noexcept { mr_.deallocate(stream_, p, n * sizeof(T)); }

  bool operator==(host_allocator const& other) const { return mr_ == other.mr_; }
  bool operator!=(host_allocator const& other) const { return !(*this == other); }

 private:
  rmm::host_async_resource_ref mr_;
  rmm::cuda_stream_view stream_;
};

// Function that returns host_device_async_resource_ref (like cudf::get_pinned_memory_resource)
rmm::host_device_async_resource_ref get_pinned_resource()
{
  static rmm::mr::pinned_host_memory_resource mr{};
  return rmm::host_device_async_resource_ref{mr};
}

// Helper to create a pinned vector (similar to cudf's make_pinned_vector_async)
template <typename T>
thrust::host_vector<T, host_allocator<T>> make_pinned_vector(std::size_t size,
                                                             rmm::cuda_stream_view stream)
{
  return thrust::host_vector<T, host_allocator<T>>(size, {get_pinned_resource(), stream});
}

// Test direct allocation through the converted ref
TEST(ResourceRefConversionAllocator, DirectAllocation)
{
  rmm::cuda_stream stream{};
  auto mr = get_pinned_resource();

  host_allocator<int> alloc{mr, stream};
  int* ptr = alloc.allocate(10);
  ASSERT_NE(ptr, nullptr);
  alloc.deallocate(ptr, 10);
}

// Test allocator copy works correctly
TEST(ResourceRefConversionAllocator, AllocatorCopy)
{
  rmm::cuda_stream stream{};
  auto mr = get_pinned_resource();

  host_allocator<int> alloc1{mr, stream};
  host_allocator<int> alloc2 = alloc1;

  int* ptr = alloc2.allocate(10);
  ASSERT_NE(ptr, nullptr);
  alloc2.deallocate(ptr, 10);
}

// Test vector construction with converted resource ref
TEST(ResourceRefConversionAllocator, VectorConstruction)
{
  rmm::cuda_stream stream{};
  thrust::host_vector<int, host_allocator<int>> vec(1, {get_pinned_resource(), stream});
  vec[0] = 42;
  ASSERT_EQ(vec[0], 42);
}

// Test vector returned from function (like cudf's make_pinned_vector pattern)
TEST(ResourceRefConversionAllocator, VectorFromFunction)
{
  rmm::cuda_stream stream{};
  auto vec = make_pinned_vector<int>(1, stream);
  vec[0]   = 42;
  ASSERT_EQ(vec[0], 42);
}

// Test vector move (exercises allocator move semantics)
TEST(ResourceRefConversionAllocator, VectorMove)
{
  rmm::cuda_stream stream{};
  thrust::host_vector<int, host_allocator<int>> vec1(1, {get_pinned_resource(), stream});
  vec1[0]   = 42;
  auto vec2 = std::move(vec1);
  ASSERT_EQ(vec2[0], 42);
}

}  // namespace
}  // namespace rmm::test
