/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/export.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <memory>
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

/**
 * @brief For a type that publicly derives from shared_resource<Impl>, extracts the
 * shared_resource<Impl> base type and provides a reference cast.
 *
 * CCCL's basic_any-based resource_ref can type-erase shared_resource<Impl> directly but not
 * classes that inherit from it. This helper extracts the base and casts to it.
 */
template <typename T, typename = void>
struct shared_resource_cast {
  static constexpr bool value = false;
};

template <typename T>
struct shared_resource_cast<
  T,
  std::void_t<
    decltype(std::declval<std::remove_cv_t<T>&>().get()),
    std::enable_if_t<std::is_base_of_v<cuda::mr::shared_resource<std::remove_reference_t<
                                         decltype(std::declval<T&>().get())>>,
                                       std::remove_cv_t<T>> and
                     not is_specialization_of_v<std::remove_cv_t<T>, cuda::mr::shared_resource>>>> {
  static constexpr bool value = true;
  using impl_type             = std::remove_reference_t<decltype(std::declval<T&>().get())>;
  using base_type             = cuda::mr::shared_resource<impl_type>;

  static base_type& cast(T& ref) noexcept { return static_cast<base_type&>(ref); }
};

// Forward declarations for use in enable_if constraints
template <typename ResourceType>
class cccl_resource_ref;

template <typename ResourceType>
class cccl_async_resource_ref;

// Traits to detect cccl_resource_ref and cccl_async_resource_ref specializations.
// Defined here (outside the class bodies) to avoid the injected-class-name issue
// where GCC resolves `cccl_async_resource_ref` to the fully-specialized type
// inside the class template body, breaking is_specialization_of_v.
template <typename T>
inline constexpr bool is_cccl_resource_ref_v = false;
template <typename R>
inline constexpr bool is_cccl_resource_ref_v<cccl_resource_ref<R>> = true;

template <typename T>
inline constexpr bool is_cccl_async_resource_ref_v = false;
template <typename R>
inline constexpr bool is_cccl_async_resource_ref_v<cccl_async_resource_ref<R>> = true;

/**
 * @brief A wrapper around CCCL synchronous_resource_ref that adds compatibility with
 * shared_resource-derived types.
 *
 * This class uses composition to wrap a CCCL resource_ref type and provides the full
 * interface of the underlying type. It enables constructing resource refs from
 * shared_resource-derived types by casting to the shared_resource base.
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
   * @brief Constructs a resource reference from a CCCL resource_ref directly.
   *
   * @param ref A CCCL resource_ref of the appropriate type
   */
  template <typename... Properties>
  cccl_resource_ref(cuda::mr::synchronous_resource_ref<Properties...> const& ref) : ref_{ref}
  {
  }

  /**
   * @brief Constructs a resource reference from a CCCL resource_ref directly (move).
   *
   * @param ref A CCCL resource_ref of the appropriate type
   */
  template <typename... Properties>
  cccl_resource_ref(cuda::mr::synchronous_resource_ref<Properties...>&& ref) : ref_{std::move(ref)}
  {
  }

  cccl_resource_ref(cccl_resource_ref const&)     = default;
  cccl_resource_ref(cccl_resource_ref&&) noexcept = default;

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
  cccl_resource_ref(cccl_resource_ref<OtherResourceType> const& other) : ref_{other.ref_}
  {
  }

  /**
   * @brief Construct a ref from a shared_resource-derived type.
   *
   * CCCL's basic_any-based resource_ref can type-erase shared_resource<T> directly but not
   * types that publicly inherit from it. This constructor casts to the shared_resource base.
   *
   * @tparam OtherResourceType A type that publicly derives from shared_resource<Impl>
   * @param other The shared_resource-derived resource to construct a ref from
   */
  template <typename OtherResourceType,
            std::enable_if_t<shared_resource_cast<OtherResourceType>::value>* = nullptr>
  cccl_resource_ref(OtherResourceType& other)
    : ref_{ResourceType{shared_resource_cast<OtherResourceType>::cast(other)}}
  {
  }

  /**
   * @brief Construct a ref from a resource.
   *
   * This constructor accepts CCCL resource types but NOT CCCL resource_ref types,
   * our own wrapper types, or shared_resource-derived types (handled by dedicated
   * constructor above).
   * The exclusions are checked FIRST to prevent recursive constraint satisfaction.
   *
   * @tparam OtherResourceType A CCCL resource type (not a resource_ref, wrapper,
   * or shared_resource)
   * @param other The resource to construct a ref from
   */
  template <
    typename OtherResourceType,
    std::enable_if_t<
      not is_specialization_of_v<std::remove_cv_t<OtherResourceType>,
                                 cuda::mr::synchronous_resource_ref> and
      not is_specialization_of_v<std::remove_cv_t<OtherResourceType>, cuda::mr::resource_ref> and
      not is_cccl_resource_ref_v<std::remove_cv_t<OtherResourceType>> and
      not is_cccl_async_resource_ref_v<std::remove_cv_t<OtherResourceType>> and
      not shared_resource_cast<OtherResourceType>::value and
      cuda::mr::synchronous_resource<OtherResourceType>>* = nullptr>
  cccl_resource_ref(OtherResourceType& other) : ref_{ResourceType{other}}
  {
  }

  cccl_resource_ref& operator=(cccl_resource_ref const&)     = default;
  cccl_resource_ref& operator=(cccl_resource_ref&&) noexcept = default;

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
  ResourceType ref_;
};

/**
 * @brief A wrapper around CCCL resource_ref (async) that adds compatibility with
 * shared_resource-derived types.
 *
 * This class is a standalone implementation (not inheriting from cccl_resource_ref)
 * to avoid recursive constraint satisfaction issues with CCCL 3.2's basic_any-based
 * resource_ref types. It provides both synchronous and asynchronous allocation methods.
 *
 * @tparam ResourceType The underlying CCCL resource_ref type (async)
 */
// Suppress spurious warning about calling a __host__ function from __host__ __device__ context
// when this class is used as a member in thrust allocators that inherit __host__ __device__
// attributes.
#ifdef __CUDACC__
#pragma nv_diagnostic push
#pragma nv_diag_suppress 20011
#endif
template <typename ResourceType>
class cccl_async_resource_ref {
 public:
  using wrapped_type = ResourceType;

  // Allow other instantiations to access our protected members for conversions
  template <typename>
  friend class cccl_async_resource_ref;

  /**
   * @brief Constructs a resource reference from a CCCL resource_ref directly.
   *
   * @param ref A CCCL resource_ref of the appropriate type
   */
  template <typename... Properties>
  cccl_async_resource_ref(cuda::mr::resource_ref<Properties...> const& ref) : ref_{ref}
  {
  }

  /**
   * @brief Constructs a resource reference from a CCCL resource_ref directly (move).
   *
   * @param ref A CCCL resource_ref of the appropriate type
   */
  template <typename... Properties>
  cccl_async_resource_ref(cuda::mr::resource_ref<Properties...>&& ref) : ref_{std::move(ref)}
  {
  }

  /**
   * @brief Constructs a resource reference from a CCCL any_resource.
   *
   * This constructor enables constructing a resource_ref from an any_resource,
   * which is useful when retrieving resources from containers that store any_resource.
   *
   * @param res A CCCL any_resource to reference
   */
  template <typename... Properties>
  cccl_async_resource_ref(cuda::mr::any_resource<Properties...>& res) : ref_{res}
  {
  }

  cccl_async_resource_ref(cccl_async_resource_ref const&)     = default;
  cccl_async_resource_ref(cccl_async_resource_ref&&) noexcept = default;

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
  template <typename OtherResourceType>
  cccl_async_resource_ref(cccl_async_resource_ref<OtherResourceType> const& other)
    : ref_{other.ref_}
  {
  }

  /**
   * @brief Construct a ref from a shared_resource-derived type.
   *
   * CCCL's basic_any-based resource_ref can type-erase shared_resource<T> directly but not
   * types that publicly inherit from it. This constructor casts to the shared_resource base.
   *
   * @tparam OtherResourceType A type that publicly derives from shared_resource<Impl>
   * @param other The shared_resource-derived resource to construct a ref from
   */
  template <typename OtherResourceType,
            std::enable_if_t<shared_resource_cast<OtherResourceType>::value>* = nullptr>
  cccl_async_resource_ref(OtherResourceType& other)
    : ref_{ResourceType{shared_resource_cast<OtherResourceType>::cast(other)}}
  {
  }

  /**
   * @brief Construct a ref from a resource.
   *
   * This constructor accepts CCCL resource types but NOT CCCL resource_ref types,
   * our own wrapper types, any_resource types, or shared_resource-derived types
   * (handled by dedicated constructor above).
   * The exclusions are checked FIRST to prevent recursive constraint satisfaction.
   *
   * @tparam OtherResourceType A CCCL resource type (not a resource_ref, wrapper, any_resource,
   * or shared_resource)
   * @param other The resource to construct a ref from
   */
  template <
    typename OtherResourceType,
    std::enable_if_t<
      not is_specialization_of_v<std::remove_cv_t<OtherResourceType>,
                                 cuda::mr::synchronous_resource_ref> and
      not is_specialization_of_v<std::remove_cv_t<OtherResourceType>, cuda::mr::resource_ref> and
      not is_specialization_of_v<std::remove_cv_t<OtherResourceType>, cuda::mr::any_resource> and
      not is_cccl_resource_ref_v<std::remove_cv_t<OtherResourceType>> and
      not is_cccl_async_resource_ref_v<std::remove_cv_t<OtherResourceType>> and
      not shared_resource_cast<OtherResourceType>::value and
      cuda::mr::resource<OtherResourceType>>* = nullptr>
  cccl_async_resource_ref(OtherResourceType& other) : ref_{ResourceType{other}}
  {
  }

  cccl_async_resource_ref& operator=(cccl_async_resource_ref const&)     = default;
  cccl_async_resource_ref& operator=(cccl_async_resource_ref&&) noexcept = default;

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

  /**
   * @brief Implicit conversion to cuda::mr::any_resource<>.
   *
   * This enables reification of the resource_ref to an owning any_resource type.
   * The conversion copies the underlying resource into the any_resource.
   */
  template <typename... Properties>
  operator cuda::mr::any_resource<Properties...>() const
  {
    return cuda::mr::any_resource<Properties...>{ref_};
  }

 protected:
  ResourceType ref_;
};
#ifdef __CUDACC__
#pragma nv_diagnostic pop
#endif

}  // namespace detail
}  // namespace RMM_NAMESPACE
