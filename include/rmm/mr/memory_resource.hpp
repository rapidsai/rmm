/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cstddef>
#include <utility>

namespace rmm {

namespace mr {

/**
 * @brief Memory kind that a memory resource can allocate
 */
enum class memory_kind {
  /// Ordinary host memory
  host      = 0,

  /// Page-locked host memory, which can be accessed by device
  pinned    = 1,

  /// GPU memory
  device    = 2,

  /// Managed memory, which is automatically migrated between host and device
  managed   = 3,
};

enum class allocation_order {
  /**
   * @brief Host-ordered allocation
   *
   * The memory is usable as soon as allocate function returns.
   * The memory must no longer be in use when calling deallocate.
   */
  host    = 0,
  /**
   * @brief Stream-ordered allocation
   *
   * The memory can be immediately accessed on a stream associated with the allocation.
   * Host code must wait until all work that was scheduled on the associated stream prior
   * to the call to allocate is complete in order to safely access the allocated memory.
   * Deallocation is scheduled in stream order. Pending work on the associated stream can
   * still use the memory.
   * The memory deallocated in stream order can be immediately recycled by allocations
   * requested for the same stream.
   */
  stream  = 1,
};


/**
 * @brief Default allocation alignment, in bytes, for given memory kind.
 */
template <memory_kind kind>
static constexpr size_t default_alignment = kind == memory_kind::host ? alignof(std::max_align_t) : 256;


/**
 * @brief Base class for all host and single-stream RMM memory resources.
 *
 * @tparam kind   The kind of memory
 * @tparam order  Determines the allocation order (host or stream)
 *
 * This is based on `std::pmr::memory_resource`:
 * https://en.cppreference.com/w/cpp/memory/memory_resource
 *
 * This class serves as the interface for all memory resources that do not need
 * to work with multiple CUDA streams.
 *
 * There are two private, pure virtual functions that all derived classes must
 * implement: `do_allocate` and `do_deallocate`. Optionally, derived classes may
 * also override `is_equal`. By default, `is_equal` simply performs an identity
 * comparison.
 *
 * The public, non-virtual functions `allocate`, `deallocate`, and `is_equal`
 * simply call the private virtual functions. The reason for this is to allow
 * implementing shared, default behavior in the base class. For example, the
 * base class' `allocate` function may log every allocation, no matter what
 * derived class implementation is used.
 */
template <memory_kind _kind, allocation_order _order = allocation_order::host>
class memory_resource {
 public:
  virtual ~memory_resource() = default;

  static constexpr mr::memory_kind kind = _kind;
  static constexpr mr::allocation_order order = _order;

  /**
   * @brief Allocates memory of size at least `bytes` bytes.
   *
   * The returned storage is aligned to the specified `alignment` if supported,
   * and to the default alignment for given backend otherwise.
   *
   * The memory can be used immediately in any context if host order is used
   * or on the associated stream if stream order is used.
   *
   * @throws std::bad_alloc When the requested `bytes` and `alignment` cannot be
   * allocated.
   *
   * @param bytes The size of the allocation
   * @param alignment Alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate(std::size_t bytes, std::size_t alignment = default_alignment<kind>)
  {
    return do_allocate(bytes, alignment);
  }

  /**
   * @brief Deallocate memory pointed to by `p`.
   *
   * `p` must have been returned by a prior call to `allocate(bytes, alignment)`
   * on a `memory_resource<kind>` that compares equal to `*this`, and the storage
   * it points to must not yet have been deallocated, otherwise behavior is
   * undefined.
   *
   * The memory region deallocated by this function.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param alignment Alignment of the allocation. This must be equal to the
   *value of `alignment` that was passed to the `allocate` call that returned
   *`p`.
   * @param stream Stream on which to perform deallocation
   */
  void deallocate(void* p, std::size_t bytes, std::size_t alignment = default_alignment<kind>)
  {
    do_deallocate(p, bytes, alignment);
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
  bool is_equal(memory_resource<kind> const& other) const noexcept { return do_is_equal(other); }

 private:
  /**---------------------------------------------------------------------------*
   * @brief Allocates memory on the host of size at least `bytes` bytes.
   *
   * The returned storage is aligned to the specified `alignment` if supported,
   * and to `alignof(std::max_align_t)` otherwise.
   *
   * @throws std::bad_alloc When the requested `bytes` and `alignment` cannot be
   * allocated.
   *
   * @param bytes The size of the allocation
   * @param alignment Alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   *---------------------------------------------------------------------------**/
  virtual void* do_allocate(std::size_t bytes,
                            std::size_t alignment = alignof(std::max_align_t)) = 0;

  /**---------------------------------------------------------------------------*
   * @brief Deallocate memory pointed to by `p`.
   *
   * `p` must have been returned by a prior call to `allocate(bytes,alignment)`
   * on a `host_memory_resource` that compares equal to `*this`, and the storage
   * it points to must not yet have been deallocated, otherwise behavior is
   * undefined.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param alignment Alignment of the allocation. This must be equal to the
   *value of `alignment` that was passed to the `allocate` call that returned
   *`p`.
   * @param stream Stream on which to perform deallocation
   *---------------------------------------------------------------------------**/
  virtual void do_deallocate(void* p,
                             std::size_t bytes,
                             std::size_t alignment = alignof(std::max_align_t)) = 0;

  /**---------------------------------------------------------------------------*
   * @brief Compare this resource to another.
   *
   * Two host_memory_resources compare equal if and only if memory allocated
   * from one host_memory_resource can be deallocated from the other and vice
   * versa.
   *
   * By default, simply checks if \p *this and \p other refer to the same
   * object, i.e., does not check if they are two objects of the same class.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   *---------------------------------------------------------------------------**/
  virtual bool do_is_equal(memory_resource const& other) const noexcept
  {
    return this == &other;
  }
};

template <memory_kind kind, allocation_order order>
constexpr mr::memory_kind memory_resource<kind, order>::kind;

template <memory_kind kind, allocation_order order>
constexpr mr::allocation_order memory_resource<kind, order>::order;

/**
 * @brief Base class for all multi-stream RMM memory resources.
 *
 * @tparam kind   The kind of memory
 *
 * This is based on `std::pmr::memory_resource`:
 * https://en.cppreference.com/w/cpp/memory/memory_resource
 *
 * When C++17 is available for use in RMM, `rmm::host_memory_resource` should
 * inherit from `std::pmr::memory_resource`.
 *
 * This class serves as the interface for all memory resources that do not need
 * to work with multiple CUDA streams.
 *
 * There are two private, pure virtual functions that all derived classes must
 * implement: `do_allocate` and `do_deallocate`. Optionally, derived classes may
 * also override `is_equal`. By default, `is_equal` simply performs an identity
 * comparison.
 *
 * The public, non-virtual functions `allocate`, `deallocate`, and `is_equal`
 * simply call the private virtual functions. The reason for this is to allow
 * implementing shared, default behavior in the base class. For example, the
 * base class' `allocate` function may log every allocation, no matter what
 * derived class implementation is used.
 */
template <memory_kind _kind>
class stream_aware_memory_resource : public memory_resource<_kind> {
 public:
  using memory_resource<_kind>::kind;
  using memory_resource<_kind>::order;

  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer is aligned at the defautl alignment for the memory kind
   * allocated by this resource.
   *
   * The memory returned must be available for immediate use on given stream.
   * If an implementation does not support streams, it should return memory which
   * can be used on any stream.
   *
   * @throws `rmm::bad_alloc` When the requested `bytes` cannot be allocated on
   * the specified `stream`.
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate_async(std::size_t bytes, cuda_stream_view stream)
  {
    return allocate_async(bytes, default_alignment<kind>, stream);
  }

  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * If supported, this operation may optionally be executed on a stream.
   * Otherwise, the stream is ignored and the null stream is used.
   *
   * The memory returned must be available for immediate use on given stream.
   * If an implementation does not support streams, it should return memory which
   * can be used on any stream.
   *
   * @throws `rmm::bad_alloc` When the requested `bytes` cannot be allocated on
   * the specified `stream`.
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate_async(std::size_t bytes, std::size_t alignment, cuda_stream_view stream)
  {
    return do_allocate_async(bytes, alignment, stream);
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * `p` must have been returned by a prior call to `allocate(bytes)` or
   * `allocate_async(bytes, stream)` on a `stream_aware_memory_resource`
   * that compares equal to `*this`, and the storage it points to must not yet
   * have been deallocated, otherwise behavior is undefined.
   *
   * The memory returned must be available for immediate use on given stream.
   * If an implementation does not support streams, it should return memory which
   * can be used on any stream.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   */
  void deallocate_async(void* p, std::size_t bytes, cuda_stream_view stream)
  {
    deallocate_async(p, bytes, default_alignment<kind>, stream);
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * `p` must have been returned by a prior call to `allocate(bytes, alignment)` or
   * `allocate_async(bytes, alignment, stream)` on a `stream_aware_memory_resource`
   * that compares equal to `*this`, and the storage it points to must not yet
   * have been deallocated, otherwise behavior is undefined.
   *
   * The memory returned must be available for immediate use on given stream.
   * If an implementation does not support streams, it should return memory which
   * can be used on any stream.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param alignment The alignment of the allocation, as was passed to the
   * call to `allocate` that returned `p`.
   * @param stream Stream on which to perform deallocation
   */
  void deallocate_async(void* p, std::size_t bytes, std::size_t alignment, cuda_stream_view stream)
  {
    do_deallocate_async(p, bytes, alignment, stream);
  }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   *
   * @return bool true if the resource supports get_mem_info, false otherwise.
   */
  virtual bool supports_get_mem_info() const noexcept = 0;

  /**
   * @brief Queries the amount of free and total memory for the resource.
   *
   * @param stream the stream whose memory manager we want to retrieve
   *
   * @returns a std::pair<size_t,size_t> which contains free memory in bytes
   * in .first and total amount of memory in .second
   */
  std::pair<std::size_t, std::size_t> get_mem_info(cuda_stream_view stream) const
  {
    return do_get_mem_info(stream);
  }

 private:
  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will be aligned at least to the required alignment.
   *
   * The memory returned must be available for immediate use on given stream.
   * If an implementation does not support streams, it should return memory which
   * can be used on any stream.
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   */
  virtual void* do_allocate_async(std::size_t bytes, std::size_t alignment, cuda_stream_view stream) = 0;

  /**
   * @brief Implements host-syncrhonous allocation by using default stream
   */
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    cuda_stream_view stream = {};
    stream.synchronize();
    return do_allocate_async(bytes, alignment, stream);
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * The returned pointer will be aligned at least to the required alignment.
   *
   * The memory returned must be available for immediate use on given stream.
   * If an implementation does not support streams, it should return memory which
   * can be used on any stream.
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   */
  virtual void do_deallocate_async(void* p, std::size_t bytes, std::size_t alignment, cuda_stream_view stream) = 0;

  /**
   * @brief Implements host-syncrhonous deallocation by using default stream
   */
  void do_deallocate(void *p, std::size_t bytes, std::size_t alignment) override {
    cuda_stream_view stream = {};
    stream.synchronize();
    return do_deallocate_async(p, bytes, alignment, stream);
  }

  /**
   * @brief Get free and available memory for memory resource
   *
   * @throws std::runtime_error if we could not get free / total memory
   *
   * @param stream the stream being executed on
   * @return std::pair with available and free memory for resource
   */
  virtual std::pair<std::size_t, std::size_t> do_get_mem_info(cuda_stream_view stream) const = 0;
};
}  // namespace mr
}  // namespace rmm
