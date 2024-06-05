/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/nvtx/ranges.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <utility>

namespace rmm::mr {
/**
 * @addtogroup device_memory_resources
 * @{
 * @file
 */

/**
 * @brief Base class for all libcudf device memory allocation.
 *
 * This class serves as the interface that all custom device memory
 * implementations must satisfy.
 *
 * There are two private, pure virtual functions that all derived classes must implement:
 *`do_allocate` and `do_deallocate`. Optionally, derived classes may also override `is_equal`. By
 * default, `is_equal` simply performs an identity comparison.
 *
 * The public, non-virtual functions `allocate`, `deallocate`, and `is_equal` simply call the
 * private virtual functions. The reason for this is to allow implementing shared, default behavior
 * in the base class. For example, the base class' `allocate` function may log every allocation, no
 * matter what derived class implementation is used.
 *
 * The `allocate` and `deallocate` APIs and implementations provide stream-ordered memory
 * allocation. This allows optimizations such as re-using memory deallocated on the same stream
 * without the overhead of stream synchronization.
 *
 * A call to `allocate(bytes, stream_a)` (on any derived class) returns a pointer that is valid to
 * use on `stream_a`. Using the memory on a different stream (say `stream_b`) is Undefined Behavior
 * unless the two streams are first synchronized, for example by using
 * `cudaStreamSynchronize(stream_a)` or by recording a CUDA event on `stream_a` and then
 * calling `cudaStreamWaitEvent(stream_b, event)`.
 *
 * The stream specified to deallocate() should be a stream on which it is valid to use the
 * deallocated memory immediately for another allocation. Typically this is the stream on which the
 * allocation was *last* used before the call to deallocate(). The passed stream may be used
 * internally by a device_memory_resource for managing available memory with minimal
 * synchronization, and it may also be synchronized at a later time, for example using a call to
 * `cudaStreamSynchronize()`.
 *
 * For this reason, it is Undefined Behavior to destroy a CUDA stream that is passed to
 * deallocate(). If the stream on which the allocation was last used has been destroyed before
 * calling deallocate() or it is known that it will be destroyed, it is likely better to synchronize
 * the stream (before destroying it) and then pass a different stream to deallocate() (e.g. the
 * default stream).
 *
 * A device_memory_resource should only be used when the active CUDA device is the same device
 * that was active when the device_memory_resource was created. Otherwise behavior is undefined.
 *
 * Creating a device_memory_resource for each device requires care to set the current device
 * before creating each resource, and to maintain the lifetime of the resources as long as they
 * are set as per-device resources. Here is an example loop that creates `unique_ptr`s to
 * pool_memory_resource objects for each device and sets them as the per-device resource for that
 * device.
 *
 * @code{.cpp}
 * using pool_mr = rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>;
 * std::vector<unique_ptr<pool_mr>> per_device_pools;
 * for(int i = 0; i < N; ++i) {
 *   cudaSetDevice(i);
 *   // Note: for brevity, omitting creation of upstream and computing initial_size
 *   per_device_pools.push_back(std::make_unique<pool_mr>(upstream, initial_size));
 *   set_per_device_resource(cuda_device_id{i}, &per_device_pools.back());
 * }
 * @endcode
 */
class device_memory_resource {
 public:
  device_memory_resource()                              = default;
  virtual ~device_memory_resource()                     = default;
  device_memory_resource(device_memory_resource const&) = default;  ///< @default_copy_constructor
  device_memory_resource(device_memory_resource&&) noexcept =
    default;  ///< @default_move_constructor
  device_memory_resource& operator=(device_memory_resource const&) =
    default;  ///< @default_copy_assignment{device_memory_resource}
  device_memory_resource& operator=(device_memory_resource&&) noexcept =
    default;  ///< @default_move_assignment{device_memory_resource}

  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * If supported, this operation may optionally be executed on a stream.
   * Otherwise, the stream is ignored and the null stream is used.
   *
   * @throws rmm::bad_alloc When the requested `bytes` cannot be allocated on
   * the specified @p stream.
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate(std::size_t bytes, cuda_stream_view stream = cuda_stream_view{})
  {
    RMM_FUNC_RANGE();
    return do_allocate(bytes, stream);
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * `p` must have been returned by a prior call to `allocate(bytes, stream)` on
   * a `device_memory_resource` that compares equal to `*this`, and the storage
   * it points to must not yet have been deallocated, otherwise behavior is
   * undefined.
   *
   * If supported, this operation may optionally be executed on a stream.
   * Otherwise, the stream is ignored and the null stream is used.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   */
  void deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream = cuda_stream_view{})
  {
    RMM_FUNC_RANGE();
    do_deallocate(ptr, bytes, stream);
  }

  /**
   * @brief Compare this resource to another.
   *
   * Two device_memory_resources compare equal if and only if memory allocated
   * from one device_memory_resource can be deallocated from the other and vice
   * versa.
   *
   * By default, simply checks if \p *this and \p other refer to the same
   * object, i.e., does not check if they are two objects of the same class.
   *
   * @param other The other resource to compare to
   * @returns If the two resources are equivalent
   */
  [[nodiscard]] bool is_equal(device_memory_resource const& other) const noexcept
  {
    return do_is_equal(other);
  }

  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * @throws rmm::bad_alloc When the requested `bytes` cannot be allocated on
   * the specified `stream`.
   *
   * @param bytes The size of the allocation
   * @param alignment The expected alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate(std::size_t bytes, std::size_t alignment)
  {
    RMM_FUNC_RANGE();
    return do_allocate(rmm::align_up(bytes, alignment), cuda_stream_view{});
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * `p` must have been returned by a prior call to `allocate(bytes, stream)` on
   * a `device_memory_resource` that compares equal to `*this`, and the storage
   * it points to must not yet have been deallocated, otherwise behavior is
   * undefined.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param alignment The alignment that was passed to the `allocate` call that returned `p`
   */
  void deallocate(void* ptr, std::size_t bytes, std::size_t alignment)
  {
    RMM_FUNC_RANGE();
    do_deallocate(ptr, rmm::align_up(bytes, alignment), cuda_stream_view{});
  }

  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * @throws rmm::bad_alloc When the requested `bytes` cannot be allocated on
   * the specified `stream`.
   *
   * @param bytes The size of the allocation
   * @param alignment The expected alignment of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate_async(std::size_t bytes, std::size_t alignment, cuda_stream_view stream)
  {
    RMM_FUNC_RANGE();
    return do_allocate(rmm::align_up(bytes, alignment), stream);
  }

  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * @throws rmm::bad_alloc When the requested `bytes` cannot be allocated on
   * the specified `stream`.
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate_async(std::size_t bytes, cuda_stream_view stream)
  {
    RMM_FUNC_RANGE();
    return do_allocate(bytes, stream);
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * `p` must have been returned by a prior call to `allocate(bytes, stream)` on
   * a `device_memory_resource` that compares equal to `*this`, and the storage
   * it points to must not yet have been deallocated, otherwise behavior is
   * undefined.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param alignment The alignment that was passed to the `allocate` call that returned `p`
   * @param stream Stream on which to perform allocation
   */
  void deallocate_async(void* ptr,
                        std::size_t bytes,
                        std::size_t alignment,
                        cuda_stream_view stream)
  {
    RMM_FUNC_RANGE();
    do_deallocate(ptr, rmm::align_up(bytes, alignment), stream);
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * `p` must have been returned by a prior call to `allocate(bytes, stream)` on
   * a `device_memory_resource` that compares equal to `*this`, and the storage
   * it points to must not yet have been deallocated, otherwise behavior is
   * undefined.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform allocation
   */
  void deallocate_async(void* ptr, std::size_t bytes, cuda_stream_view stream)
  {
    RMM_FUNC_RANGE();
    do_deallocate(ptr, bytes, stream);
  }

  /**
   * @brief Comparison operator with another device_memory_resource
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equivalent
   */
  [[nodiscard]] bool operator==(device_memory_resource const& other) const noexcept
  {
    return do_is_equal(other);
  }

  /**
   * @brief Comparison operator with another device_memory_resource
   *
   * @param other The other resource to compare to
   * @return false If the two resources are equivalent
   * @return true If the two resources are not equivalent
   */
  [[nodiscard]] bool operator!=(device_memory_resource const& other) const noexcept
  {
    return !do_is_equal(other);
  }

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `device_memory_resource` provides device accessible memory
   */
  friend void get_property(device_memory_resource const&, cuda::mr::device_accessible) noexcept {}

 private:
  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * If supported, this operation may optionally be executed on a stream.
   * Otherwise, the stream is ignored and the null stream is used.
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   */
  virtual void* do_allocate(std::size_t bytes, cuda_stream_view stream) = 0;

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * If supported, this operation may optionally be executed on a stream.
   * Otherwise, the stream is ignored and the null stream is used.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   */
  virtual void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) = 0;

  /**
   * @brief Compare this resource to another.
   *
   * Two device_memory_resources compare equal if and only if memory allocated
   * from one device_memory_resource can be deallocated from the other and vice
   * versa.
   *
   * By default, simply checks if \p *this and \p other refer to the same
   * object, i.e., does not check if they are two objects of the same class.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  [[nodiscard]] virtual bool do_is_equal(device_memory_resource const& other) const noexcept
  {
    return this == &other;
  }
};
static_assert(cuda::mr::async_resource_with<device_memory_resource, cuda::mr::device_accessible>);
/** @} */  // end of group
}  // namespace rmm::mr
