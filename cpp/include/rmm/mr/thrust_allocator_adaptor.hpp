/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/detail/cuda_memory_resource.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/detail/thrust_namespace.h>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/std/limits>
#include <thrust/device_ptr.h>
#include <thrust/device_reference.h>
#include <thrust/memory.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/memory_resource.h>

#include <cstddef>
#include <type_traits>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resource_adaptors
 * @{
 * @file
 */

/**
 * @brief A Thrust-compatible memory resource that wraps an RMM device_async_resource_ref.
 *
 * This class adapts RMM's stream-ordered allocation interface to work with Thrust's
 * memory resource system. Each instance is bound to a specific stream and resource.
 *
 * @note This class is marked `final` as required by `thrust::mr::allocator`.
 */
class rmm_thrust_device_resource final
  : public thrust::mr::memory_resource<thrust::device_ptr<void>> {
 public:
  /**
   * @brief Construct a memory resource with a stream and upstream resource.
   *
   * @param stream The stream to use for allocations
   * @param mr The upstream RMM resource
   * @param device The CUDA device associated with this resource
   */
  rmm_thrust_device_resource(cuda_stream_view stream,
                             rmm::device_async_resource_ref mr,
                             cuda_device_id device)
    : _stream{stream}, _mr{mr}, _device{device}
  {
  }

  /**
   * @brief Allocates device memory.
   *
   * @param bytes The number of bytes to allocate
   * @param alignment The alignment (unused, RMM uses 256-byte alignment)
   * @return A device pointer to the allocated memory
   */
  pointer do_allocate(std::size_t bytes, [[maybe_unused]] std::size_t alignment) override
  {
    cuda_set_device_raii dev{_device};
    return thrust::device_ptr<void>{_mr.allocate(_stream, bytes)};
  }

  /**
   * @brief Deallocates device memory.
   *
   * @param ptr The pointer to deallocate
   * @param bytes The number of bytes to deallocate
   * @param alignment The alignment (unused)
   */
  void do_deallocate(pointer ptr,
                     std::size_t bytes,
                     [[maybe_unused]] std::size_t alignment) override
  {
    cuda_set_device_raii dev{_device};
    _mr.deallocate(_stream, thrust::raw_pointer_cast(ptr), bytes);
  }

  /**
   * @briefreturn{The stream used for allocations}
   */
  [[nodiscard]] cuda_stream_view stream() const noexcept { return _stream; }

  /**
   * @briefreturn{The upstream RMM resource}
   */
  [[nodiscard]] rmm::device_async_resource_ref resource() const noexcept { return _mr; }

  /**
   * @briefreturn{The CUDA device associated with this resource}
   */
  [[nodiscard]] cuda_device_id device() const noexcept { return _device; }

 private:
  cuda_stream_view _stream;            ///< Stream for allocations
  rmm::device_async_resource_ref _mr;  ///< Upstream RMM resource
  cuda_device_id _device;              ///< Associated CUDA device
};

/**
 * @brief Helper base class to ensure rmm_thrust_device_resource is initialized
 * before thrust::mr::allocator (which needs a pointer to it).
 *
 * In C++, base classes are initialized before members in declaration order.
 * By inheriting from this class first, we ensure _resource exists before
 * thrust::mr::allocator's constructor receives &_resource.
 */
struct thrust_allocator_resource_holder {
  rmm_thrust_device_resource _resource;  ///< The memory resource instance

  /**
   * @brief Constructs the holder with a resource initialized for the given stream and device.
   *
   * @param stream The stream for allocations
   * @param mr The upstream RMM resource
   * @param device The CUDA device
   */
  thrust_allocator_resource_holder(cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr,
                                   cuda_device_id device)
    : _resource{stream, mr, device}
  {
  }
};

/**
 * @brief An `allocator` compatible with Thrust containers and algorithms using
 * a `device_async_resource_ref` for memory (de)allocation.
 *
 * Unlike a `device_async_resource_ref`, `thrust_allocator` is typed and bound to
 * allocate objects of a specific type `T`, but can be freely rebound to other
 * types.
 *
 * The allocator records the current CUDA device and may only be used with a backing
 * `device_async_resource_ref` valid for the same device.
 *
 * This allocator uses `thrust::mr::allocator` as its base class, which requires
 * a `thrust::mr::memory_resource`. An internal `rmm_thrust_device_resource` adapts
 * RMM's stream-ordered allocation to Thrust's memory resource interface.
 *
 * @tparam T The type of the objects that will be allocated by this allocator
 */
template <typename T>
class thrust_allocator : private thrust_allocator_resource_holder,
                         public thrust::mr::allocator<T, rmm_thrust_device_resource> {
 public:
  using base_type = thrust::mr::allocator<T, rmm_thrust_device_resource>;  ///< Base allocator type
  using pointer   = typename base_type::pointer;    ///< Pointer type returned by allocate
  using size_type = typename base_type::size_type;  ///< Type used for allocation sizes

  /**
   * @brief Provides the type of a `thrust_allocator` instantiated with another
   * type.
   *
   * @tparam U the other type to use for instantiation
   */
  template <typename U>
  struct rebind {
    using other = thrust_allocator<U>;  ///< The type to bind to
  };

  /**
   * @brief Default constructor creates an allocator using the default memory
   * resource and default stream.
   */
  thrust_allocator()
    : thrust_allocator_resource_holder{cuda_stream_view{},
                                       rmm::mr::get_current_device_resource_ref(),
                                       get_current_cuda_device()},
      base_type{&_resource}
  {
  }

  /**
   * @brief Constructs a `thrust_allocator` using the default device memory
   * resource and specified stream.
   *
   * @param stream The stream to be used for device memory (de)allocation
   */
  explicit thrust_allocator(cuda_stream_view stream)
    : thrust_allocator_resource_holder{stream,
                                       rmm::mr::get_current_device_resource_ref(),
                                       get_current_cuda_device()},
      base_type{&_resource}
  {
  }

  /**
   * @brief Constructs a `thrust_allocator` using a device memory resource and
   * stream.
   *
   * @param stream The stream to be used for device memory (de)allocation
   * @param mr The resource to be used for device memory allocation
   */
  thrust_allocator(cuda_stream_view stream, rmm::device_async_resource_ref mr)
    : thrust_allocator_resource_holder{stream, mr, get_current_cuda_device()}, base_type{&_resource}
  {
  }

  /**
   * @brief Copy constructor. Copies the resource pointer and stream.
   *
   * @param other The `thrust_allocator` to copy
   */
  thrust_allocator(thrust_allocator const& other)
    : thrust_allocator_resource_holder{other._resource.stream(),
                                       other._resource.resource(),
                                       other._resource.device()},
      base_type{&_resource}
  {
  }

  /**
   * @brief Copy constructor from allocator of different type. Copies the resource pointer and
   * stream.
   *
   * @param other The `thrust_allocator` to copy
   */
  template <typename U>
  thrust_allocator(thrust_allocator<U> const& other)
    : thrust_allocator_resource_holder{other.stream(), other.resource(), other.device()},
      base_type{&_resource}
  {
  }

  /**
   * @brief Copy assignment operator.
   *
   * @param other The `thrust_allocator` to copy
   * @return Reference to this allocator
   */
  thrust_allocator& operator=(thrust_allocator const& other)
  {
    if (this != &other) {
      _resource = rmm_thrust_device_resource{
        other._resource.stream(), other._resource.resource(), other._resource.device()};
      // base_type stores a pointer to _resource, which remains valid
    }
    return *this;
  }

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept
  {
    return _resource.resource();
  }

  /**
   * @briefreturn{The stream used by this allocator}
   */
  [[nodiscard]] cuda_stream_view stream() const noexcept { return _resource.stream(); }

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   * @deprecated Use get_upstream_resource() instead
   */
  [[nodiscard]] rmm::device_async_resource_ref resource() const noexcept
  {
    return _resource.resource();
  }

  /**
   * @briefreturn{The CUDA device associated with this allocator}
   */
  [[nodiscard]] cuda_device_id device() const noexcept { return _resource.device(); }

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `thrust_allocator` provides device accessible memory
   */
  friend void get_property(thrust_allocator const&, cuda::mr::device_accessible) noexcept {}

  template <typename U>
  friend class thrust_allocator;
};
/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
