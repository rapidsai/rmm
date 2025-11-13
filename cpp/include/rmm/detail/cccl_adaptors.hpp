/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/cuda_memory_resource.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/device_memory_resource_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>

namespace RMM_NAMESPACE {

namespace detail {

// Helper base class to hold the view (Base from Member idiom)
// This is initialized before the main base class, allowing us to pass it to the base constructor
struct view_holder {
  view_holder() = default;  // Default constructor for inherited base constructors
  view_holder(rmm::mr::device_memory_resource* ptr) : view_{ptr} {}
  rmm::mr::detail::device_memory_resource_view view_;
};

template <typename ResourceType>
class cccl_resource_ref : private view_holder, public ResourceType {
 public:
  using base = ResourceType;

  using base::base;

  cccl_resource_ref(base const& other) : base(other) {}

  cccl_resource_ref(base&& other) : base(std::move(other)) {}

  /**
   * @brief Constructs a resource reference from a raw `device_memory_resource` pointer.
   *
   * This constructor enables compatibility with CCCL 3.2 by wrapping the pointer in a
   * `device_memory_resource_view`, which is copyable unlike the virtual base class.
   *
   * @param ptr Non-null pointer to a `device_memory_resource`
   */
  cccl_resource_ref(rmm::mr::device_memory_resource* ptr)
    : view_holder(ptr), base(view_holder::view_)
  {
  }

  /**
   * @brief Constructs a resource reference from a `device_memory_resource` reference.
   *
   * This constructor enables compatibility with CCCL 3.2 by wrapping the address in a
   * `device_memory_resource_view`, which is copyable unlike the virtual base class.
   *
   * @param res Reference to a `device_memory_resource`
   */
  cccl_resource_ref(rmm::mr::device_memory_resource& res)
    : view_holder(&res), base(view_holder::view_)
  {
  }

  void* allocate_sync(std::size_t bytes) { return base::allocate_sync(bytes); }

  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    return base::allocate_sync(bytes, alignment);
  }

  void deallocate_sync(void* ptr, std::size_t bytes) noexcept
  {
    return base::deallocate_sync(ptr, bytes);
  }

  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    return base::deallocate_sync(ptr, bytes, alignment);
  }
};

template <typename ResourceType>
class cccl_async_resource_ref : private view_holder, public ResourceType {
 public:
  using base = ResourceType;

  using base::base;

  cccl_async_resource_ref(base const& other) : base(other) {}
  cccl_async_resource_ref(base&& other) : base(std::move(other)) {}

  /**
   * @brief Constructs an async resource reference from a raw `device_memory_resource` pointer.
   *
   * This constructor enables compatibility with CCCL 3.2 by wrapping the pointer in a
   * `device_memory_resource_view`, which is copyable unlike the virtual base class.
   *
   * @param ptr Non-null pointer to a `device_memory_resource`
   */
  cccl_async_resource_ref(rmm::mr::device_memory_resource* ptr)
    : view_holder(ptr), base(view_holder::view_)
  {
  }

  /**
   * @brief Constructs an async resource reference from a `device_memory_resource` reference.
   *
   * This constructor enables compatibility with CCCL 3.2 by wrapping the address in a
   * `device_memory_resource_view`, which is copyable unlike the virtual base class.
   *
   * @param res Reference to a `device_memory_resource`
   */
  cccl_async_resource_ref(rmm::mr::device_memory_resource& res)
    : view_holder(&res), base(view_holder::view_)
  {
  }

  void* allocate_sync(std::size_t bytes) { return base::allocate_sync(bytes); }

  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    return base::allocate_sync(bytes, alignment);
  }

  void deallocate_sync(void* ptr, std::size_t bytes) noexcept
  {
    return base::deallocate_sync(ptr, bytes);
  }

  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    return base::deallocate_sync(ptr, bytes, alignment);
  }

  void* allocate(cuda_stream_view stream, std::size_t bytes)
  {
    return base::allocate(stream, bytes);
  }

  void* allocate(cuda_stream_view stream, std::size_t bytes, std::size_t alignment)
  {
    return base::allocate(stream, bytes, alignment);
  }

  void deallocate(cuda_stream_view stream, void* ptr, std::size_t bytes) noexcept
  {
    return base::deallocate(stream, ptr, bytes);
  }

  void deallocate(cuda_stream_view stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    return base::deallocate(stream, ptr, bytes, alignment);
  }
};

}  // namespace detail
}  // namespace RMM_NAMESPACE
