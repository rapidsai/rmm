/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Tests for resource_ref type conversions between host_device and device/host variants

#include <rmm/cuda_stream.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

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

/*
// Test conversion from host_device_async_resource_ref to host_async_resource_ref
TEST(ResourceRefConversion, HostDeviceToHostAsync)
{
  rmm::mr::pinned_host_memory_resource mr{};

  // Create a host_device_async_resource_ref
  rmm::host_device_async_resource_ref hd_ref{mr};

  // Convert to host_async_resource_ref
  rmm::host_async_resource_ref h_ref{hd_ref};

  // Use the converted ref
  rmm::cuda_stream stream{};
  void* ptr = h_ref.allocate(stream, 1024);
  ASSERT_NE(ptr, nullptr);
  h_ref.deallocate(stream, ptr, 1024);
}

// Test conversion from host_device_resource_ref to device_resource_ref (sync version)
TEST(ResourceRefConversion, HostDeviceToDeviceSync)
{
  rmm::mr::pinned_host_memory_resource mr{};

  // Create a host_device_resource_ref
  rmm::host_device_resource_ref hd_ref{mr};

  // Convert to device_resource_ref
  rmm::device_resource_ref d_ref{hd_ref};

  // Use the converted ref
  void* ptr = d_ref.allocate_sync(1024);
  ASSERT_NE(ptr, nullptr);
  d_ref.deallocate_sync(ptr, 1024);
}

// Test conversion from host_device_resource_ref to host_resource_ref (sync version)
TEST(ResourceRefConversion, HostDeviceToHostSync)
{
  rmm::mr::pinned_host_memory_resource mr{};

  // Create a host_device_resource_ref
  rmm::host_device_resource_ref hd_ref{mr};

  // Convert to host_resource_ref
  rmm::host_resource_ref h_ref{hd_ref};

  // Use the converted ref
  void* ptr = h_ref.allocate_sync(1024);
  ASSERT_NE(ptr, nullptr);
  h_ref.deallocate_sync(ptr, 1024);
}

// Test that the converted ref still works after the original goes out of scope
TEST(ResourceRefConversion, ConvertedRefOutlivesOriginal)
{
  rmm::mr::pinned_host_memory_resource mr{};
  rmm::cuda_stream stream{};

  rmm::device_async_resource_ref d_ref{rmm::host_device_async_resource_ref{mr}};

  // Use the converted ref - the original host_device_async_resource_ref is now gone
  void* ptr = d_ref.allocate(stream, 1024);
  ASSERT_NE(ptr, nullptr);
  d_ref.deallocate(stream, ptr, 1024);
}

// Test copy of converted ref
TEST(ResourceRefConversion, CopyOfConvertedRef)
{
  rmm::mr::pinned_host_memory_resource mr{};
  rmm::cuda_stream stream{};

  rmm::host_device_async_resource_ref hd_ref{mr};
  rmm::device_async_resource_ref d_ref1{hd_ref};

  // Copy the converted ref
  rmm::device_async_resource_ref d_ref2 = d_ref1;

  // Both should work
  void* ptr1 = d_ref1.allocate(stream, 512);
  void* ptr2 = d_ref2.allocate(stream, 512);
  ASSERT_NE(ptr1, nullptr);
  ASSERT_NE(ptr2, nullptr);
  d_ref1.deallocate(stream, ptr1, 512);
  d_ref2.deallocate(stream, ptr2, 512);
}

// Test move of converted ref
TEST(ResourceRefConversion, MoveOfConvertedRef)
{
  rmm::mr::pinned_host_memory_resource mr{};
  rmm::cuda_stream stream{};

  rmm::host_device_async_resource_ref hd_ref{mr};
  rmm::device_async_resource_ref d_ref1{hd_ref};

  // Move the converted ref
  rmm::device_async_resource_ref d_ref2 = std::move(d_ref1);

  // Only d_ref2 should be used after move
  void* ptr = d_ref2.allocate(stream, 1024);
  ASSERT_NE(ptr, nullptr);
  d_ref2.deallocate(stream, ptr, 1024);
}
*/
