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

#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

namespace RMM_NAMESPACE {

namespace detail {

// Helper base class to hold the view (Base from Member idiom)
// This is initialized before the main base class, allowing us to pass it to the base constructor
struct view_holder {
  view_holder() = default;
  view_holder(rmm::mr::device_memory_resource* ptr) : view_{ptr} {}
  std::optional<rmm::mr::detail::device_memory_resource_view> view_;
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
    : view_holder{ptr}, base{*view_holder::view_}
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
    : view_holder{&res}, base{*view_holder::view_}
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
   *
   * If the view is present (e.g., when constructed from device_memory_resource*), we reconstruct
   * the base from our local view. Otherwise, we copy the base directly.
   */
  cccl_resource_ref(cccl_resource_ref const& other)
    : view_holder{static_cast<view_holder const&>(other)},
      base{view_holder::view_.has_value() ? base{*view_holder::view_}
                                          : static_cast<base const&>(other)}
  {
  }

  /**
   * @brief Move constructor that properly reconstructs the base to point to the new view.
   *
   * If the view is present (e.g., when constructed from device_memory_resource*), we reconstruct
   * the base from our local view. Otherwise, we move the base directly.
   */
  cccl_resource_ref(cccl_resource_ref&& other) noexcept
    : view_holder{static_cast<view_holder&&>(other)},
      base{view_holder::view_.has_value() ? base{*view_holder::view_} : static_cast<base&&>(other)}
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
   *
   * If the view is present, we reconstruct the base from our local view.
   * Otherwise, we copy the base directly.
   */
  cccl_resource_ref& operator=(cccl_resource_ref const& other)
  {
    if (this != &other) {
      view_holder::view_ = other.view_;
      base::operator=(view_holder::view_.has_value() ? base(*view_holder::view_)
                                                     : static_cast<base const&>(other));
    }
    return *this;
  }

  /**
   * @brief Move assignment operator.
   *
   * If the view is present, we reconstruct the base from our local view.
   * Otherwise, we move the base directly.
   */
  cccl_resource_ref& operator=(cccl_resource_ref&& other) noexcept
  {
    if (this != &other) {
      view_holder::view_ = std::move(other.view_);
      base::operator=(view_holder::view_.has_value() ? base(*view_holder::view_)
                                                     : static_cast<base&&>(other));
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
    : view_holder{ptr}, base{*view_holder::view_}
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
    : view_holder{&res}, base{*view_holder::view_}
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
   * @brief Constructs an async resource reference from a CCCL any_resource.
   *
   * This constructor enables interoperability with CCCL 3.2 any_resource types,
   * allowing RMM async resource_ref types to be constructed from any_resource.
   * The any_resource is implicitly converted to a resource_ref.
   *
   * @tparam Properties The properties of the any_resource
   * @param any_res A CCCL any_resource with compatible properties
   */
  template <typename... Properties>
  cccl_async_resource_ref(cuda::mr::any_resource<Properties...>& any_res)
    : view_holder(), base(any_res)
  {
  }

  /**
   * @brief Constructs an async resource reference from a const CCCL any_resource.
   *
   * This constructor enables interoperability with CCCL 3.2 any_resource types,
   * allowing RMM async resource_ref types to be constructed from const any_resource.
   * The any_resource is implicitly converted to a resource_ref.
   *
   * @note Uses const_cast because resource_ref requires a non-const lvalue,
   *       but creating a reference doesn't modify the resource.
   *
   * @tparam Properties The properties of the any_resource
   * @param any_res A const CCCL any_resource with compatible properties
   */
  template <typename... Properties>
  cccl_async_resource_ref(cuda::mr::any_resource<Properties...> const& any_res)
    : view_holder(), base(const_cast<cuda::mr::any_resource<Properties...>&>(any_res))
  {
  }

  /**
   * @brief Copy constructor that properly reconstructs the base to point to the new view.
   *
   * The implicit copy constructor would copy the view_holder correctly, but the base
   * would still point to the original object's view. We need to reconstruct the base
   * to point to our own view.
   *
   * If the view is present (e.g., when constructed from device_memory_resource*), we reconstruct
   * the base from our local view. Otherwise, we copy the base directly.
   */
  cccl_async_resource_ref(cccl_async_resource_ref const& other)
    : view_holder{static_cast<view_holder const&>(other)},
      base{view_holder::view_.has_value() ? base{*view_holder::view_}
                                          : static_cast<base const&>(other)}
  {
  }

  /**
   * @brief Move constructor that properly reconstructs the base to point to the new view.
   *
   * If the view is present (e.g., when constructed from device_memory_resource*), we reconstruct
   * the base from our local view. Otherwise, we move the base directly.
   */
  cccl_async_resource_ref(cccl_async_resource_ref&& other) noexcept
    : view_holder{static_cast<view_holder&&>(other)},
      base{view_holder::view_.has_value() ? base{*view_holder::view_} : static_cast<base&&>(other)}
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
   *
   * If the view is present, we reconstruct the base from our local view.
   * Otherwise, we copy the base directly.
   */
  cccl_async_resource_ref& operator=(cccl_async_resource_ref const& other)
  {
    if (this != &other) {
      view_holder::view_ = other.view_;
      base::operator=(view_holder::view_.has_value() ? base(*view_holder::view_)
                                                     : static_cast<base const&>(other));
    }
    return *this;
  }

  /**
   * @brief Move assignment operator.
   *
   * If the view is present, we reconstruct the base from our local view.
   * Otherwise, we move the base directly.
   */
  cccl_async_resource_ref& operator=(cccl_async_resource_ref&& other) noexcept
  {
    if (this != &other) {
      view_holder::view_ = std::move(other.view_);
      base::operator=(view_holder::view_.has_value() ? base(*view_holder::view_)
                                                     : static_cast<base&&>(other));
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
