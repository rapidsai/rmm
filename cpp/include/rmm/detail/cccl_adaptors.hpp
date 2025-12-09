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

/**
 * @brief Helper trait to check if a type is a specialization of a class template.
 *
 * Used to exclude CCCL resource_ref types from resource-accepting constructors.
 */
template <class Type, template <class...> class Template>
inline constexpr bool is_specialization_of_v = false;

template <template <class...> class Template, class... Args>
inline constexpr bool is_specialization_of_v<Template<Args...>, Template> = true;

// Forward declarations for use in enable_if constraints
template <typename ResourceType>
class cccl_resource_ref;

template <typename ResourceType>
class cccl_async_resource_ref;

/**
 * @brief A wrapper around CCCL synchronous_resource_ref that adds compatibility with
 * device_memory_resource pointers.
 *
 * This class uses composition to wrap a CCCL resource_ref type and provides the full
 * interface of the underlying type. It enables constructing resource refs from
 * device_memory_resource pointers by wrapping them in a device_memory_resource_view.
 *
 * @tparam ResourceType The underlying CCCL synchronous_resource_ref type
 */
template <typename ResourceType>
class cccl_resource_ref {
 public:
  using wrapped_type = ResourceType;

  // Allow other instantiations to access our protected members for conversions
  template <typename>
  friend class cccl_resource_ref;

  /**
   * @brief Constructs a resource reference from a raw `device_memory_resource` pointer.
   *
   * This constructor enables compatibility with CCCL 3.2 by wrapping the pointer in a
   * `device_memory_resource_view`, which is copyable unlike the virtual base class.
   *
   * @param ptr Non-null pointer to a `device_memory_resource`
   */
  cccl_resource_ref(rmm::mr::device_memory_resource* ptr) : view_{ptr}, ref_{*view_} {}

  /**
   * @brief Constructs a resource reference from a `device_memory_resource` reference.
   *
   * This constructor enables compatibility with CCCL 3.2 by wrapping the address in a
   * `device_memory_resource_view`, which is copyable unlike the virtual base class.
   *
   * @param res Reference to a `device_memory_resource`
   */
  cccl_resource_ref(rmm::mr::device_memory_resource& res) : view_{&res}, ref_{*view_} {}

  /**
   * @brief Constructs a resource reference from a CCCL resource_ref directly.
   *
   * This constructor enables interoperability with CCCL 3.2 resource_ref types,
   * allowing RMM resource_ref types to be constructed from CCCL resource_ref types.
   *
   * @param ref A CCCL resource_ref of the appropriate type
   */
  template <typename... Properties>
  cccl_resource_ref(cuda::mr::synchronous_resource_ref<Properties...> const& ref)
    : view_{std::nullopt}, ref_{ref}
  {
  }

  /**
   * @brief Constructs a resource reference from a CCCL resource_ref directly (move).
   *
   * This constructor enables interoperability with CCCL 3.2 resource_ref types,
   * allowing RMM resource_ref types to be constructed from CCCL resource_ref types
   * using move semantics.
   *
   * @param ref A CCCL resource_ref of the appropriate type
   */
  template <typename... Properties>
  cccl_resource_ref(cuda::mr::synchronous_resource_ref<Properties...>&& ref)
    : view_{std::nullopt}, ref_{std::move(ref)}
  {
  }

  /**
   * @brief Copy constructor that properly reconstructs the ref to point to the new view.
   *
   * If the view is present (e.g., when constructed from device_memory_resource*), we reconstruct
   * the ref from our local view. Otherwise, we copy the ref directly.
   */
  cccl_resource_ref(cccl_resource_ref const& other)
    : view_{other.view_}, ref_{view_.has_value() ? ResourceType{*view_} : other.ref_}
  {
  }

  /**
   * @brief Move constructor that properly reconstructs the ref to point to the new view.
   *
   * If the view is present (e.g., when constructed from device_memory_resource*), we reconstruct
   * the ref from our local view. Otherwise, we move the ref directly.
   */
  cccl_resource_ref(cccl_resource_ref&& other) noexcept
    : view_{std::move(other.view_)},
      ref_{view_.has_value() ? ResourceType{*view_} : std::move(other.ref_)}
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
  template <typename OtherResourceType>
  cccl_resource_ref(cccl_resource_ref<OtherResourceType> const& other)
    : view_{other.view_}, ref_{view_.has_value() ? ResourceType{*view_} : ResourceType{other.ref_}}
  {
  }

  /**
   * @brief Construct a ref from a resource.
   *
   * This constructor accepts CCCL resource types but NOT CCCL resource_ref types,
   * our own wrapper types, or device_memory_resource derived types. The exclusions
   * are checked FIRST to prevent recursive constraint satisfaction.
   *
   * @tparam OtherResourceType A CCCL resource type (not a resource_ref, wrapper, or DMR)
   * @param other The resource to construct a ref from
   */
  template <
    typename OtherResourceType,
    std::enable_if_t<
      not is_specialization_of_v<std::remove_cv_t<OtherResourceType>,
                                 cuda::mr::synchronous_resource_ref> and
      not is_specialization_of_v<std::remove_cv_t<OtherResourceType>, cuda::mr::resource_ref> and
      not is_specialization_of_v<std::remove_cv_t<OtherResourceType>, cccl_resource_ref> and
      not is_specialization_of_v<std::remove_cv_t<OtherResourceType>, cccl_async_resource_ref> and
      not std::is_base_of_v<rmm::mr::device_memory_resource,
                            std::remove_cv_t<OtherResourceType>> and
      cuda::mr::synchronous_resource<OtherResourceType>>* = nullptr>
  cccl_resource_ref(OtherResourceType& other) : view_{}, ref_{ResourceType{other}}
  {
  }

  /**
   * @brief Copy assignment operator.
   *
   * If the view is present, we reconstruct the ref from our local view.
   * Otherwise, we copy the ref directly.
   */
  cccl_resource_ref& operator=(cccl_resource_ref const& other)
  {
    if (this != &other) {
      view_ = other.view_;
      ref_  = view_.has_value() ? ResourceType{*view_} : other.ref_;
    }
    return *this;
  }

  /**
   * @brief Move assignment operator.
   *
   * If the view is present, we reconstruct the ref from our local view.
   * Otherwise, we move the ref directly.
   */
  cccl_resource_ref& operator=(cccl_resource_ref&& other) noexcept
  {
    if (this != &other) {
      view_ = std::move(other.view_);
      ref_  = view_.has_value() ? ResourceType{*view_} : std::move(other.ref_);
    }
    return *this;
  }

  void* allocate_sync(std::size_t bytes) { return ref_.allocate_sync(bytes); }

  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    return ref_.allocate_sync(bytes, alignment);
  }

  void deallocate_sync(void* ptr, std::size_t bytes) noexcept
  {
    return ref_.deallocate_sync(ptr, bytes);
  }

  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    return ref_.deallocate_sync(ptr, bytes, alignment);
  }

  /**
   * @brief Returns the type_info of the wrapped resource.
   */
  [[nodiscard]] auto type() const noexcept -> decltype(std::declval<ResourceType const&>().type())
  {
    return ref_.type();
  }

  /**
   * @brief Equality comparison operator.
   */
  friend bool operator==(cccl_resource_ref const& lhs, cccl_resource_ref const& rhs) noexcept
  {
    return lhs.ref_ == rhs.ref_;
  }

  /**
   * @brief Inequality comparison operator.
   */
  friend bool operator!=(cccl_resource_ref const& lhs, cccl_resource_ref const& rhs) noexcept
  {
    return !(lhs == rhs);
  }

  /**
   * @brief Forwards a property query to the wrapped resource_ref.
   */
  template <typename Property>
  friend auto constexpr get_property(cccl_resource_ref const& ref, Property prop) noexcept
    -> decltype(get_property(std::declval<ResourceType const&>(), prop))
  {
    return get_property(ref.ref_, prop);
  }

  /**
   * @brief Attempts to get a property from the wrapped resource_ref.
   */
  template <typename Property>
  friend auto try_get_property(cccl_resource_ref const& ref, Property prop) noexcept
    -> decltype(try_get_property(std::declval<ResourceType const&>(), prop))
  {
    return try_get_property(ref.ref_, prop);
  }

 protected:
  std::optional<rmm::mr::detail::device_memory_resource_view> view_;
  ResourceType ref_;
};

/**
 * @brief A wrapper around CCCL resource_ref (async) that adds compatibility with
 * device_memory_resource pointers.
 *
 * This class is a standalone implementation (not inheriting from cccl_resource_ref)
 * to avoid recursive constraint satisfaction issues with CCCL 3.2's basic_any-based
 * resource_ref types. It provides both synchronous and asynchronous allocation methods.
 *
 * @tparam ResourceType The underlying CCCL resource_ref type (async)
 */
template <typename ResourceType>
class cccl_async_resource_ref {
 public:
  using wrapped_type = ResourceType;

  // Allow other instantiations to access our protected members for conversions
  template <typename>
  friend class cccl_async_resource_ref;

  /**
   * @brief Constructs a resource reference from a raw `device_memory_resource` pointer.
   *
   * This constructor enables compatibility with CCCL 3.2 by wrapping the pointer in a
   * `device_memory_resource_view`, which is copyable unlike the virtual base class.
   *
   * @param ptr Non-null pointer to a `device_memory_resource`
   */
  cccl_async_resource_ref(rmm::mr::device_memory_resource* ptr) : view_{ptr}, ref_{*view_} {}

  /**
   * @brief Constructs a resource reference from a `device_memory_resource` reference.
   *
   * This constructor enables compatibility with CCCL 3.2 by wrapping the address in a
   * `device_memory_resource_view`, which is copyable unlike the virtual base class.
   *
   * @param res Reference to a `device_memory_resource`
   */
  cccl_async_resource_ref(rmm::mr::device_memory_resource& res) : view_{&res}, ref_{*view_} {}

  /**
   * @brief Constructs a resource reference from a CCCL resource_ref directly.
   *
   * This constructor enables interoperability with CCCL 3.2 resource_ref types,
   * allowing RMM resource_ref types to be constructed from CCCL resource_ref types.
   *
   * @param ref A CCCL resource_ref of the appropriate type
   */
  template <typename... Properties>
  cccl_async_resource_ref(cuda::mr::resource_ref<Properties...> const& ref)
    : view_{std::nullopt}, ref_{ref}
  {
  }

  /**
   * @brief Constructs a resource reference from a CCCL resource_ref directly (move).
   *
   * This constructor enables interoperability with CCCL 3.2 resource_ref types,
   * allowing RMM resource_ref types to be constructed from CCCL resource_ref types
   * using move semantics.
   *
   * @param ref A CCCL resource_ref of the appropriate type
   */
  template <typename... Properties>
  cccl_async_resource_ref(cuda::mr::resource_ref<Properties...>&& ref)
    : view_{std::nullopt}, ref_{std::move(ref)}
  {
  }

  /**
   * @brief Copy constructor that properly reconstructs the ref to point to the new view.
   *
   * If the view is present (e.g., when constructed from device_memory_resource*), we reconstruct
   * the ref from our local view. Otherwise, we copy the ref directly.
   */
  cccl_async_resource_ref(cccl_async_resource_ref const& other)
    : view_{other.view_}, ref_{view_.has_value() ? ResourceType{*view_} : other.ref_}
  {
  }

  /**
   * @brief Move constructor that properly reconstructs the ref to point to the new view.
   *
   * If the view is present (e.g., when constructed from device_memory_resource*), we reconstruct
   * the ref from our local view. Otherwise, we move the ref directly.
   */
  cccl_async_resource_ref(cccl_async_resource_ref&& other) noexcept
    : view_{std::move(other.view_)},
      ref_{view_.has_value() ? ResourceType{*view_} : std::move(other.ref_)}
  {
  }

  /**
   * @brief Conversion constructor from a cccl_async_resource_ref with a convertible ResourceType.
   *
   * This enables conversions like host_device_async_resource_ref -> device_async_resource_ref,
   * where the source type has a superset of properties compared to the target type.
   * The underlying CCCL resource_ref types handle the actual property compatibility check.
   *
   * IMPORTANT: This constructor must copy the view_ from the source to preserve the
   * device_memory_resource pointer. Without this, the converted resource_ref will have
   * an empty view_, causing corrupt pointer dereferences during deallocation.
   *
   * @tparam OtherResourceType A CCCL async resource_ref type that is convertible to ResourceType
   * @param other The source async resource_ref to convert from
   */
  template <typename OtherResourceType>
  cccl_async_resource_ref(cccl_async_resource_ref<OtherResourceType> const& other)
    : view_{other.view_}, ref_{view_.has_value() ? ResourceType{*view_} : ResourceType{other.ref_}}
  {
  }

  /**
   * @brief Construct a ref from a resource.
   *
   * This constructor accepts CCCL resource types but NOT CCCL resource_ref types,
   * our own wrapper types, or device_memory_resource derived types. The exclusions
   * are checked FIRST to prevent recursive constraint satisfaction.
   *
   * @tparam OtherResourceType A CCCL resource type (not a resource_ref, wrapper, or DMR)
   * @param other The resource to construct a ref from
   */
  template <
    typename OtherResourceType,
    std::enable_if_t<
      not is_specialization_of_v<std::remove_cv_t<OtherResourceType>,
                                 cuda::mr::synchronous_resource_ref> and
      not is_specialization_of_v<std::remove_cv_t<OtherResourceType>, cuda::mr::resource_ref> and
      not is_specialization_of_v<std::remove_cv_t<OtherResourceType>, cccl_resource_ref> and
      not is_specialization_of_v<std::remove_cv_t<OtherResourceType>, cccl_async_resource_ref> and
      not std::is_base_of_v<rmm::mr::device_memory_resource,
                            std::remove_cv_t<OtherResourceType>> and
      cuda::mr::resource<OtherResourceType>>* = nullptr>
  cccl_async_resource_ref(OtherResourceType& other) : view_{}, ref_{ResourceType{other}}
  {
  }

  /**
   * @brief Copy assignment operator.
   *
   * If the view is present, we reconstruct the ref from our local view.
   * Otherwise, we copy the ref directly.
   */
  cccl_async_resource_ref& operator=(cccl_async_resource_ref const& other)
  {
    if (this != &other) {
      view_ = other.view_;
      ref_  = view_.has_value() ? ResourceType{*view_} : other.ref_;
    }
    return *this;
  }

  /**
   * @brief Move assignment operator.
   *
   * If the view is present, we reconstruct the ref from our local view.
   * Otherwise, we move the ref directly.
   */
  cccl_async_resource_ref& operator=(cccl_async_resource_ref&& other) noexcept
  {
    if (this != &other) {
      view_ = std::move(other.view_);
      ref_  = view_.has_value() ? ResourceType{*view_} : std::move(other.ref_);
    }
    return *this;
  }

  // Synchronous allocation methods (delegated to the underlying ref)
  void* allocate_sync(std::size_t bytes) { return ref_.allocate_sync(bytes); }

  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    return ref_.allocate_sync(bytes, alignment);
  }

  void deallocate_sync(void* ptr, std::size_t bytes) noexcept
  {
    return ref_.deallocate_sync(ptr, bytes);
  }

  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    return ref_.deallocate_sync(ptr, bytes, alignment);
  }

  // Asynchronous allocation methods
  void* allocate(cuda_stream_view stream, std::size_t bytes)
  {
    return ref_.allocate(stream, bytes);
  }

  void* allocate(cuda_stream_view stream, std::size_t bytes, std::size_t alignment)
  {
    return ref_.allocate(stream, bytes, alignment);
  }

  void deallocate(cuda_stream_view stream, void* ptr, std::size_t bytes) noexcept
  {
    return ref_.deallocate(stream, ptr, bytes);
  }

  void deallocate(cuda_stream_view stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    return ref_.deallocate(stream, ptr, bytes, alignment);
  }

  /**
   * @brief Returns the type_info of the wrapped resource.
   */
  [[nodiscard]] auto type() const noexcept -> decltype(std::declval<ResourceType const&>().type())
  {
    return ref_.type();
  }

  /**
   * @brief Equality comparison operator.
   */
  friend bool operator==(cccl_async_resource_ref const& lhs,
                         cccl_async_resource_ref const& rhs) noexcept
  {
    return lhs.ref_ == rhs.ref_;
  }

  /**
   * @brief Inequality comparison operator.
   */
  friend bool operator!=(cccl_async_resource_ref const& lhs,
                         cccl_async_resource_ref const& rhs) noexcept
  {
    return !(lhs == rhs);
  }

  /**
   * @brief Forwards a property query to the wrapped resource_ref.
   */
  template <typename Property>
  friend auto constexpr get_property(cccl_async_resource_ref const& ref, Property prop) noexcept
    -> decltype(get_property(std::declval<ResourceType const&>(), prop))
  {
    return get_property(ref.ref_, prop);
  }

  /**
   * @brief Attempts to get a property from the wrapped resource_ref.
   */
  template <typename Property>
  friend auto try_get_property(cccl_async_resource_ref const& ref, Property prop) noexcept
    -> decltype(try_get_property(std::declval<ResourceType const&>(), prop))
  {
    return try_get_property(ref.ref_, prop);
  }

 protected:
  std::optional<rmm::mr::detail::device_memory_resource_view> view_;
  ResourceType ref_;
};

}  // namespace detail
}  // namespace RMM_NAMESPACE
