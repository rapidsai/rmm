/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/detail/exec_check_disable.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/detail/thrust_namespace.h>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/memory.h>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resource_adaptors
 * @{
 * @file
 */
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
 * @tparam T The type of the objects that will be allocated by this allocator
 */
template <typename T>
class thrust_allocator : public thrust::device_malloc_allocator<T> {
 public:
  using Base      = thrust::device_malloc_allocator<T>;  ///< The base type of this allocator
  using pointer   = typename Base::pointer;              ///< The pointer type
  using size_type = typename Base::size_type;            ///< The size type

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
  RMM_EXEC_CHECK_DISABLE
  thrust_allocator() {}

  /**
   * @brief Constructs a `thrust_allocator` using the default device memory
   * resource and specified stream.
   *
   * @param stream The stream to be used for device memory (de)allocation
   */
  RMM_EXEC_CHECK_DISABLE
  explicit thrust_allocator(cuda_stream_view stream) : _stream{stream} {}

  /**
   * @brief Constructs a `thrust_allocator` using a device memory resource and
   * stream.
   *
   * @param mr The resource to be used for device memory allocation
   * @param stream The stream to be used for device memory (de)allocation
   */
  RMM_EXEC_CHECK_DISABLE
  thrust_allocator(cuda_stream_view stream, cuda::mr::any_resource<cuda::mr::device_accessible> mr)
    : _stream{stream}, _mr(std::move(mr))
  {
  }

  /**
   * @brief Copy constructor. Copies the resource pointer and stream.
   *
   * @param other The `thrust_allocator` to copy
   */
  RMM_EXEC_CHECK_DISABLE
  thrust_allocator(thrust_allocator const& other)
    : Base(other), _stream{other._stream}, _mr(other._mr), _device{other._device}
  {
  }

  /**
   * @brief Move constructor. Moves the resource pointer and stream.
   *
   * @param other The `thrust_allocator` to move from
   */
  RMM_EXEC_CHECK_DISABLE
  thrust_allocator(thrust_allocator&& other) noexcept
    : Base(std::move(other)),
      _stream{other._stream},
      _mr(std::move(other._mr)),
      _device{other._device}
  {
  }

  /// @default_copy_assignment{thrust_allocator}
  thrust_allocator& operator=(thrust_allocator const&) = default;
  /// @default_move_assignment{thrust_allocator}
  thrust_allocator& operator=(thrust_allocator&&) noexcept = default;

  /**
   * @brief Copy constructor from a `thrust_allocator` of a different type.
   * Copies the resource pointer and stream.
   *
   * @param other The `thrust_allocator` to copy
   */
  RMM_EXEC_CHECK_DISABLE
  template <typename U>
  thrust_allocator(thrust_allocator<U> const& other)
    : _mr(other.resource()), _stream{other.stream()}, _device{other._device}
  {
  }

  /**
   * @brief Allocate objects of type `T`
   *
   * @param num  The number of elements of type `T` to allocate
   * @return pointer Pointer to the newly allocated storage
   */
  pointer allocate(size_type num)
  {
    cuda_set_device_raii dev{_device};
    return thrust::device_pointer_cast(
      static_cast<T*>(_mr.allocate(_stream, num * sizeof(T), rmm::CUDA_ALLOCATION_ALIGNMENT)));
  }

  /**
   * @brief Deallocates objects of type `T`
   *
   * @param ptr Pointer returned by a previous call to `allocate`
   * @param num number of elements, *must* be equal to the argument passed to the
   * prior `allocate` call that produced `ptr`
   */
  void deallocate(pointer ptr, size_type num) noexcept
  {
    cuda_set_device_raii dev{_device};
    return _mr.deallocate(
      _stream, thrust::raw_pointer_cast(ptr), num * sizeof(T), rmm::CUDA_ALLOCATION_ALIGNMENT);
  }

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept
  {
    return rmm::device_async_resource_ref{_mr};
  }

  /**
   * @briefreturn{The stream used by this allocator}
   */
  [[nodiscard]] cuda_stream_view stream() const noexcept { return _stream; }

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `thrust_allocator` provides device accessible memory
   */
  RMM_CONSTEXPR_FRIEND void get_property(thrust_allocator const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 private:
  cuda::stream_ref _stream{cudaStream_t{nullptr}};
  mutable cuda::mr::any_resource<cuda::mr::device_accessible> _mr{
    rmm::mr::get_current_device_resource_ref()};
  cuda_device_id _device{get_current_cuda_device()};
};
/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
