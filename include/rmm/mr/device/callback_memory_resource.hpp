/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
 * @brief Callback function type used by callback memory resource for allocation.
 *
 * The signature of the callback function is:
 *   `void* allocate_callback_t(std::size_t bytes, cuda_stream_view stream, void* arg);
 *
 * `bytes` specifies the number of bytes to allocate. `stream` is the CUDA stream
 * on which the memory is used. Additional information to the callback may be passed
 * via `arg`. The function returns the pointer to the allocated memory.
 */
using allocate_callback_t = std::function<void*(std::size_t, cuda_stream_view, void*)>;

/**
 * @brief Callback function type used by callback_memory_resource for deallocation.
 *
 * The signature of the callback function is:
 *   `void* deallocate_callback_t(void* ptr, std::size_t bytes, cuda_stream_view stream, void* arg);
 *
 * `ptr` specifies the pointer to the memory to be freed, while `bytes` specified
 * the number of bytes to free. `stream` is the CUDA stream on which the memory is used.
 * Additional information to the callback may be passed via `arg`
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
   * @throws Nothing.
   *
   * @param allocate_callback The callback function used for allocation
   * @param deallocate_callback The callback function used for deallocation
   */
  callback_memory_resource(allocate_callback_t allocate_callback,
                           deallocate_callback_t deallocate_callback,
                           void* allocate_callback_arg,
                           void* deallocate_callback_arg) noexcept
    : allocate_callback_(allocate_callback),
      deallocate_callback_(deallocate_callback),
      allocate_callback_arg_(allocate_callback_arg),
      deallocate_callback_arg_(deallocate_callback_arg)
  {
  }

  callback_memory_resource()                                = delete;
  ~callback_memory_resource() override                      = default;
  callback_memory_resource(callback_memory_resource const&) = delete;
  callback_memory_resource& operator=(callback_memory_resource const&) = delete;
  callback_memory_resource(callback_memory_resource&&) noexcept        = default;
  callback_memory_resource& operator=(callback_memory_resource&&) noexcept = default;

 private:
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
  {
    return allocate_callback_(bytes, stream, allocate_callback_arg_);
  }

  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) override
  {
    deallocate_callback_(ptr, bytes, stream, deallocate_callback_arg_);
  }

  [[nodiscard]] std::pair<std::size_t, std::size_t> do_get_mem_info(cuda_stream_view) const override
  {
    throw std::runtime_error("cannot get free / total memory");
  }

  [[nodiscard]] virtual bool supports_streams() const noexcept { return false; }
  [[nodiscard]] virtual bool supports_get_mem_info() const noexcept { return false; }

  allocate_callback_t allocate_callback_;
  deallocate_callback_t deallocate_callback_;
  void* allocate_callback_arg_;
  void* deallocate_callback_arg_;
};

}  // namespace rmm::mr
