/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda/memory_resource>
#include <cuda/std/cstddef>

#include <cstddef>
#include <optional>
#include <type_traits>

// A simple memory resource for testing
struct test_memory_resource {
  void* allocate(cuda::stream_ref,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    return ::operator new(bytes, std::align_val_t{alignment});
  }

  void deallocate(cuda::stream_ref,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t))
  {
    ::operator delete(ptr, bytes, std::align_val_t{alignment});
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    return allocate_sync(bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes) { return allocate_sync(bytes, alignof(std::max_align_t)); }

  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment)
  {
    deallocate_sync(ptr, bytes, alignment);
  }

  void deallocate_sync(void* ptr, std::size_t bytes)
  {
    deallocate_sync(ptr, bytes, alignof(std::max_align_t));
  }

  bool operator==(test_memory_resource const&) const { return true; }
  bool operator!=(test_memory_resource const&) const { return false; }

  friend void get_property(test_memory_resource const&, cuda::mr::device_accessible) noexcept {}
  friend void get_property(test_memory_resource const&, cuda::mr::host_accessible) noexcept {}
};

static_assert(cuda::mr::resource<test_memory_resource>);
static_assert(cuda::mr::synchronous_resource<test_memory_resource>);

// Wrapper class similar to RMM's cccl_async_resource_ref
template <typename ResourceType>
class cccl_async_resource_ref {
 public:
  using wrapped_type = ResourceType;

  template <typename>
  friend class cccl_async_resource_ref;

  // Constructor from a resource
  template <typename Resource>
  cccl_async_resource_ref(Resource& res) : ref_{res}
  {
  }

  // Copy constructor
  cccl_async_resource_ref(cccl_async_resource_ref const& other) : ref_{other.ref_} {}

  // Move constructor
  cccl_async_resource_ref(cccl_async_resource_ref&& other) noexcept : ref_{std::move(other.ref_)} {}

  // Conversion constructor from wrapper with different ResourceType
  // THIS IS THE PROBLEMATIC CONSTRUCTOR - it triggers recursive constraint satisfaction
  template <typename OtherResourceType>
  cccl_async_resource_ref(cccl_async_resource_ref<OtherResourceType> const& other)
    : ref_{other.ref_}
  {
  }

  // Copy assignment
  cccl_async_resource_ref& operator=(cccl_async_resource_ref const& other)
  {
    if (this != &other) { ref_ = other.ref_; }
    return *this;
  }

  // Move assignment
  cccl_async_resource_ref& operator=(cccl_async_resource_ref&& other) noexcept
  {
    if (this != &other) { ref_ = std::move(other.ref_); }
    return *this;
  }

  void* allocate(cuda::stream_ref stream, std::size_t bytes)
  {
    return ref_.allocate(stream, bytes);
  }

  void deallocate(cuda::stream_ref stream, void* ptr, std::size_t bytes) noexcept
  {
    return ref_.deallocate(stream, ptr, bytes);
  }

  friend bool operator==(cccl_async_resource_ref const& lhs,
                         cccl_async_resource_ref const& rhs) noexcept
  {
    return lhs.ref_ == rhs.ref_;
  }

  friend bool operator!=(cccl_async_resource_ref const& lhs,
                         cccl_async_resource_ref const& rhs) noexcept
  {
    return !(lhs == rhs);
  }

 private:
  ResourceType ref_;
};

// Type aliases similar to RMM's
using device_async_resource_ref =
  cccl_async_resource_ref<cuda::mr::resource_ref<cuda::mr::device_accessible>>;

using host_async_resource_ref =
  cccl_async_resource_ref<cuda::mr::resource_ref<cuda::mr::host_accessible>>;

using host_device_async_resource_ref = cccl_async_resource_ref<
  cuda::mr::resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>>;

// This static_assert triggers the recursive constraint satisfaction error in C++20
// Comment it out to see the error move to the test function below
static_assert(
  std::is_constructible_v<device_async_resource_ref, host_device_async_resource_ref>,
  "device_async_resource_ref must be constructible from host_device_async_resource_ref");

void test_conversion()
{
  test_memory_resource mr{};

  // Create a host_device_async_resource_ref
  host_device_async_resource_ref hd_ref{mr};

  // Convert to device_async_resource_ref - this also triggers the error
  device_async_resource_ref d_ref{hd_ref};

  // Use it
  void* ptr = d_ref.allocate(cuda::stream_ref{}, 1024);
  d_ref.deallocate(cuda::stream_ref{}, ptr, 1024);
}

int main()
{
  test_conversion();
  return 0;
}
