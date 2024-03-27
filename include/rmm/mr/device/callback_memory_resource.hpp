/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <rmm/mr/device/device_memory_resource.hpp>

#include <cstddef>
#include <functional>
#include <utility>

namespace rmm::mr {
/**
 * @addtogroup device_memory_resources
 * @{
 * @file
 */

/**
 * @brief Callback function type used by callback memory resource for allocation.
 *
 * The signature of the callback function is:
 *   `void* allocate_callback_t(std::size_t bytes, cuda_stream_view stream, void* arg);
 *
 * * Returns a pointer to an allocation of at least `bytes` usable immediately on
 *   `stream`. The stream-ordered behavior requirements are identical to
 *   `device_memory_resource::allocate`.
 *
 * * This signature is compatible with `do_allocate` but adds the extra function
 *   parameter `arg`. The `arg` is provided to the constructor of the
 *   `callback_memory_resource` and will be forwarded along to every invocation
 *   of the callback function.
 */
using allocate_callback_t = std::function<void*(std::size_t, cuda_stream_view, void*)>;

/**
 * @brief Callback function type used by callback_memory_resource for deallocation.
 *
 * The signature of the callback function is:
 *   `void deallocate_callback_t(void* ptr, std::size_t bytes, cuda_stream_view stream, void* arg);
 *
 * * Deallocates memory pointed to by `ptr`. `bytes` specifies the size of the allocation
 *   in bytes, and must equal the value of `bytes` that was passed to the allocate callback
 *   function. The stream-ordered behavior requirements are identical to
 *   `device_memory_resource::deallocate`.
 *
 * * This signature is compatible with `do_deallocate` but adds the extra function
 *   parameter `arg`. The `arg` is provided to the constructor of the
 *   `callback_memory_resource` and will be forwarded along to every invocation
 *   of the callback function.
 */
using deallocate_callback_t = std::function<void(void*, std::size_t, cuda_stream_view, void*)>;

/**
 * @brief A device memory resource that uses the provided callbacks for memory allocation
 * and deallocation.
 */
class callback_memory_resource final : public device_memory_resource {
 public:
  /**
   * @brief Construct a new callback memory resource.
   *
   * Constructs a callback memory resource that uses the user-provided callbacks
   * `allocate_callback` for allocation and `deallocate_callback` for deallocation.
   *
   * @param allocate_callback The callback function used for allocation
   * @param deallocate_callback The callback function used for deallocation
   * @param allocate_callback_arg Additional context passed to `allocate_callback`.
   * It is the caller's responsibility to maintain the lifetime of the pointed-to data
   * for the duration of the lifetime of the `callback_memory_resource`.
   * @param deallocate_callback_arg Additional context passed to `deallocate_callback`.
   * It is the caller's responsibility to maintain the lifetime of the pointed-to data
   * for the duration of the lifetime of the `callback_memory_resource`.
   */
  callback_memory_resource(allocate_callback_t allocate_callback,
                           deallocate_callback_t deallocate_callback,
                           void* allocate_callback_arg   = nullptr,
                           void* deallocate_callback_arg = nullptr) noexcept
    : allocate_callback_(allocate_callback),
      deallocate_callback_(deallocate_callback),
      allocate_callback_arg_(allocate_callback_arg),
      deallocate_callback_arg_(deallocate_callback_arg)
  {
  }

  callback_memory_resource()                                           = delete;
  ~callback_memory_resource() override                                 = default;
  callback_memory_resource(callback_memory_resource const&)            = delete;
  callback_memory_resource& operator=(callback_memory_resource const&) = delete;
  callback_memory_resource(callback_memory_resource&&) noexcept =
    default;  ///< @default_move_constructor
  callback_memory_resource& operator=(callback_memory_resource&&) noexcept =
    default;  ///< @default_move_assignment{callback_memory_resource}

 private:
  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * If supported by the callback, this operation may optionally be executed on
   * a stream.  Otherwise, the stream is ignored and the null stream is used.
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
  {
    return allocate_callback_(bytes, stream, allocate_callback_arg_);
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * If supported by the callback, this operation may optionally be executed on
   * a stream.  Otherwise, the stream is ignored and the null stream is used.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   */
  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) override
  {
    deallocate_callback_(ptr, bytes, stream, deallocate_callback_arg_);
  }

  allocate_callback_t allocate_callback_;
  deallocate_callback_t deallocate_callback_;
  void* allocate_callback_arg_;
  void* deallocate_callback_arg_;
};

/** @} */  // end of group
}  // namespace rmm::mr
