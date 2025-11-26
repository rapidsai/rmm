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
  view_holder() = default;
  view_holder(rmm::mr::device_memory_resource* ptr) : view_{ptr} {}
  rmm::mr::detail::device_memory_resource_view view_;
};

template <typename ResourceType>
class cccl_resource_ref : private view_holder, public ResourceType {
 public:
  using base = ResourceType;

  /**
   * @brief Constructs a resource reference from a raw `device_memory_resource` pointer.
   *
   * This constructor enables compatibility with CCCL 3.2 by wrapping the pointer in a
   * `device_memory_resource_view`, which is copyable unlike the virtual base class.
   *
   * @param ptr Non-null pointer to a `device_memory_resource`
   */
  cccl_resource_ref(rmm::mr::device_memory_resource* ptr)
    : view_holder{ptr}, base{view_holder::view_}
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
    : view_holder{&res}, base{view_holder::view_}
  {
  }

  /**
   * @brief Constructs a resource reference from a CCCL resource_ref directly.
   *
   * This constructor enables interoperability with CCCL 3.2 resource_ref types,
   * allowing RMM resource_ref types to be constructed from CCCL resource_ref types.
   *
   * @param ref A CCCL resource_ref of the appropriate type
   */
  cccl_resource_ref(ResourceType const& ref) : view_holder{}, base{ref} {}

  /**
   * @brief Constructs a resource reference from a CCCL resource_ref directly (move).
   *
   * This constructor enables interoperability with CCCL 3.2 resource_ref types,
   * allowing RMM resource_ref types to be constructed from CCCL resource_ref types
   * using move semantics.
   *
   * @param ref A CCCL resource_ref of the appropriate type
   */
  cccl_resource_ref(ResourceType&& ref) : view_holder{}, base{std::move(ref)} {}

  /**
   * @brief Copy constructor that properly reconstructs the base to point to the new view.
   */
  cccl_resource_ref(cccl_resource_ref const& other)
    : view_holder{static_cast<view_holder const&>(other)}, base{view_holder::view_}
  {
  }

  /**
   * @brief Move constructor that properly reconstructs the base to point to the new view.
   */
  cccl_resource_ref(cccl_resource_ref&& other) noexcept
    : view_holder{static_cast<view_holder&&>(other)}, base{view_holder::view_}
  {
  }

  /**
   * @brief Conversion constructor from a cccl_resource_ref with a convertible ResourceType.
   *
   * This enables conversions like host_device_resource_ref -> device_resource_ref,
   * where the source type has a superset of properties compared to the target type.
   * The underlying CCCL resource_ref types handle the actual property compatibility check.
   *
   * @tparam OtherResourceType A CCCL resource_ref type that is convertible to ResourceType
   * @param other The source resource_ref to convert from
   */
  template <typename OtherResourceType,
            typename = std::enable_if_t<std::is_constructible_v<ResourceType, OtherResourceType>>>
  cccl_resource_ref(cccl_resource_ref<OtherResourceType> const& other)
    : view_holder{}, base{static_cast<OtherResourceType const&>(other)}
  {
  }

  /**
   * @brief Copy assignment operator.
   */
  cccl_resource_ref& operator=(cccl_resource_ref const& other)
  {
    if (this != &other) {
      view_holder::view_ = other.view_;
      base::operator=(base(view_holder::view_));
    }
    return *this;
  }

  /**
   * @brief Move assignment operator.
   */
  cccl_resource_ref& operator=(cccl_resource_ref&& other) noexcept
  {
    if (this != &other) {
      view_holder::view_ = std::move(other.view_);
      base::operator=(base(view_holder::view_));
    }
    return *this;
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

  /**
   * @brief Constructs an async resource reference from a raw `device_memory_resource` pointer.
   *
   * This constructor enables compatibility with CCCL 3.2 by wrapping the pointer in a
   * `device_memory_resource_view`, which is copyable unlike the virtual base class.
   *
   * @param ptr Non-null pointer to a `device_memory_resource`
   */
  cccl_async_resource_ref(rmm::mr::device_memory_resource* ptr)
    : view_holder{ptr}, base{view_holder::view_}
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
    : view_holder{&res}, base{view_holder::view_}
  {
  }

  /**
   * @brief Constructs an async resource reference from a CCCL resource_ref directly.
   *
   * This constructor enables interoperability with CCCL 3.2 resource_ref types,
   * allowing RMM async resource_ref types to be constructed from CCCL resource_ref types.
   *
   * @param ref A CCCL async resource_ref of the appropriate type
   */
  cccl_async_resource_ref(ResourceType const& ref) : view_holder{}, base{ref} {}

  /**
   * @brief Constructs an async resource reference from a CCCL resource_ref directly (move).
   *
   * This constructor enables interoperability with CCCL 3.2 resource_ref types,
   * allowing RMM async resource_ref types to be constructed from CCCL resource_ref types
   * using move semantics.
   *
   * @param ref A CCCL async resource_ref of the appropriate type
   */
  cccl_async_resource_ref(ResourceType&& ref) : view_holder{}, base{std::move(ref)} {}

  /**
   * @brief Copy constructor that properly reconstructs the base to point to the new view.
   *
   * The implicit copy constructor would copy the view_holder correctly, but the base
   * would still point to the original object's view. We need to reconstruct the base
   * to point to our own view.
   */
  cccl_async_resource_ref(cccl_async_resource_ref const& other)
    : view_holder{static_cast<view_holder const&>(other)}, base{view_holder::view_}
  {
  }

  /**
   * @brief Move constructor that properly reconstructs the base to point to the new view.
   */
  cccl_async_resource_ref(cccl_async_resource_ref&& other) noexcept
    : view_holder{static_cast<view_holder&&>(other)}, base{view_holder::view_}
  {
  }

  /**
   * @brief Conversion constructor from a cccl_async_resource_ref with a convertible ResourceType.
   *
   * This enables conversions like host_device_async_resource_ref -> device_async_resource_ref,
   * where the source type has a superset of properties compared to the target type.
   * The underlying CCCL resource_ref types handle the actual property compatibility check.
   *
   * @tparam OtherResourceType A CCCL async resource_ref type that is convertible to ResourceType
   * @param other The source async resource_ref to convert from
   */
  template <typename OtherResourceType,
            typename = std::enable_if_t<std::is_constructible_v<ResourceType, OtherResourceType>>>
  cccl_async_resource_ref(cccl_async_resource_ref<OtherResourceType> const& other)
    : view_holder{}, base{static_cast<OtherResourceType const&>(other)}
  {
  }

  /**
   * @brief Copy assignment operator.
   */
  cccl_async_resource_ref& operator=(cccl_async_resource_ref const& other)
  {
    if (this != &other) {
      view_holder::view_ = other.view_;
      base::operator=(base(view_holder::view_));
    }
    return *this;
  }

  /**
   * @brief Move assignment operator.
   */
  cccl_async_resource_ref& operator=(cccl_async_resource_ref&& other) noexcept
  {
    if (this != &other) {
      view_holder::view_ = std::move(other.view_);
      base::operator=(base(view_holder::view_));
    }
    return *this;
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
