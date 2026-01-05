/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Tests for resource_ref type conversions between host_device and device/host variants

#include <rmm/cuda_stream.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/host_vector.h>

#include <gtest/gtest.h>

class new_delete_memory_resource {
 public:
  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    try {
      return rmm::detail::aligned_host_allocate(
        bytes, alignment, [](std::size_t size) { return ::operator new(size); });
    } catch (std::bad_alloc const& e) {
      RMM_FAIL("Failed to allocate memory: " + std::string{e.what()}, rmm::out_of_memory);
    }
  }

  void* allocate([[maybe_unused]] cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate_sync(bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    rmm::detail::aligned_host_deallocate(
      ptr, bytes, alignment, [](void* ptr) { ::operator delete(ptr); });
  }

  void deallocate([[maybe_unused]] cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate_sync(ptr, bytes, alignment);
  }

  bool operator==(new_delete_memory_resource const& other) const { return true; }

  bool operator!=(new_delete_memory_resource const& other) const { return !operator==(other); }

  // NOLINTBEGIN
  friend void get_property(new_delete_memory_resource const&, cuda::mr::host_accessible) noexcept {}
  // NOLINTEND
};

TEST(ResourceRefConversion, ResourceToRef)
{
  auto mr = new_delete_memory_resource{};
  static_assert(std::is_copy_constructible_v<new_delete_memory_resource>);
  static_assert(cuda::mr::synchronous_resource<new_delete_memory_resource>);
  static_assert(
    cuda::mr::synchronous_resource_with<new_delete_memory_resource, cuda::mr::host_accessible>);
  static_assert(
    std::is_constructible_v<cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible>,
                            new_delete_memory_resource&>);
  rmm::host_resource_ref mr_ref{mr};
  // Use the converted ref
  void* ptr = mr_ref.allocate_sync(1024);
  ASSERT_NE(ptr, nullptr);
  mr_ref.deallocate_sync(ptr, 1024);
}

// Test conversion from host_device_async_resource_ref to device_async_resource_ref
TEST(ResourceRefConversion, HostDeviceToDeviceSync)
{
  rmm::mr::pinned_host_memory_resource mr{};

  // Create a host_device_async_resource_ref
  rmm::host_device_resource_ref hd_ref{mr};
  static_assert(cuda::mr::synchronous_resource_with<decltype(hd_ref), cuda::mr::host_accessible>);
  static_assert(cuda::mr::synchronous_resource_with<decltype(hd_ref), cuda::mr::device_accessible>);
  static_assert(cuda::has_property<decltype(hd_ref), cuda::mr::host_accessible>);
  static_assert(cuda::has_property<decltype(hd_ref), cuda::mr::device_accessible>);

  // Convert to device_async_resource_ref
  rmm::device_resource_ref d_ref{hd_ref};
  static_assert(!cuda::mr::synchronous_resource_with<decltype(d_ref), cuda::mr::host_accessible>);
  static_assert(cuda::mr::synchronous_resource_with<decltype(d_ref), cuda::mr::device_accessible>);
  static_assert(!cuda::has_property<decltype(d_ref), cuda::mr::host_accessible>);
  static_assert(cuda::has_property<decltype(d_ref), cuda::mr::device_accessible>);

  // Use the converted ref
  void* ptr = d_ref.allocate_sync(1024);
  ASSERT_NE(ptr, nullptr);
  d_ref.deallocate_sync(ptr, 1024);
}

TEST(ResourceRefConversion, HostDeviceToDeviceAsync)
{
  rmm::mr::pinned_host_memory_resource mr{};

  // Create a host_device_async_resource_ref
  rmm::host_device_async_resource_ref hd_ref{mr};

  // Convert to device_async_resource_ref
  rmm::device_async_resource_ref d_ref{hd_ref};

  // Use the converted ref
  rmm::cuda_stream stream{};
  void* ptr = d_ref.allocate(stream, 1024);
  ASSERT_NE(ptr, nullptr);
  d_ref.deallocate(stream, ptr, 1024);
}

// Host allocator that takes a resource_ref (similar to cudf's rmm_host_allocator)
template <typename T>
class host_allocator {
 public:
  using value_type = T;

  host_allocator() = delete;

  template <typename ResourceType>
  host_allocator(ResourceType mr, rmm::cuda_stream_view stream) : mr_(mr), stream_(stream)
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
