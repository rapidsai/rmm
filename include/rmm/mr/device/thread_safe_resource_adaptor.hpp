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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <mutex>

namespace rmm::mr {
/**
 * @brief Resource that adapts `Upstream` memory resource adaptor to be thread safe.
 *
 * An instance of this resource can be constructured with an existing, upstream resource in order
 * to satisfy allocation requests. This adaptor wraps allocations and deallocations from Upstream
 * in a mutex lock.
 */
class thread_safe_resource_adaptor final : public device_memory_resource {
 public:
  using lock_t = std::lock_guard<std::mutex>;

  /**
   * @brief Construct a new thread safe resource adaptor using `upstream` to satisfy
   * allocation requests.
   *
   * All allocations and frees are protected by a mutex lock
   *
   * @throws `rmm::logic_error` if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory.
   */
  thread_safe_resource_adaptor(
    cuda::stream_ordered_resource_view<cuda::memory_access::device> upstream)
    : upstream_{upstream}
  {
    RMM_EXPECTS(
      upstream != cuda::stream_ordered_resource_view<cuda::memory_access::device>{nullptr},
      "Unexpected null upstream resource pointer.");
  }

  thread_safe_resource_adaptor()                                    = delete;
  ~thread_safe_resource_adaptor() override                          = default;
  thread_safe_resource_adaptor(thread_safe_resource_adaptor const&) = delete;
  thread_safe_resource_adaptor(thread_safe_resource_adaptor&&)      = delete;
  thread_safe_resource_adaptor& operator=(thread_safe_resource_adaptor const&) = delete;
  thread_safe_resource_adaptor& operator=(thread_safe_resource_adaptor&&) = delete;

  /**
   * @brief Get the upstream memory resource.
   *
   * @return View of the upstream memory resource.
   */
  cuda::stream_ordered_resource_view<cuda::memory_access::device> get_upstream() const noexcept
  {
    return upstream_;
  }

  /**
   * @copydoc rmm::mr::device_memory_resource::supports_streams()
   */
  bool supports_streams() const noexcept override { return true; }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   *
   * @return bool true if the upstream resource supports get_mem_info, false otherwise.
   */
  bool supports_get_mem_info() const noexcept override { return false; }

 private:
  /**
   * @brief Allocates memory of size at least `bytes` using the upstream
   * resource with thread safety.
   *
   * @throws `rmm::bad_alloc` if the requested allocation could not be fulfilled
   * by the upstream resource.
   *
   * @param bytes The size, in bytes, of the allocation
   * @param stream Stream on which to perform the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
  {
    lock_t lock(mtx);
    return upstream_->allocate_async(bytes, stream.value());
  }

  /**
   * @brief Free allocation of size `bytes` pointed to to by `ptr`.s
   *
   * @throws Nothing.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes Size of the allocation
   * @param stream Stream on which to perform the deallocation
   */
  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) override
  {
    lock_t lock(mtx);
    upstream_->deallocate_async(ptr, bytes, stream.value());
  }

  /**
   * @brief Compare the upstream resource to another.
   *
   * @throws Nothing.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equivalent
   */
  bool do_is_equal(cuda::memory_resource<memory_kind> const& other) const noexcept override
  {
    if (this == &other) { return true; }
    auto const* thread_safe_other = dynamic_cast<thread_safe_resource_adaptor const*>(&other);
    if (thread_safe_other != nullptr) { return upstream_ == thread_safe_other->get_upstream(); }
    return upstream_ == &other;
  }

  /**
   * @brief Get free and available memory from upstream resource.
   *
   * @throws `rmm::cuda_error` if unable to retrieve memory info.
   *
   * @param stream Stream on which to get the mem info.
   * @return std::pair contaiing free_size and total_size of memory
   */
  std::pair<std::size_t, std::size_t> do_get_mem_info(cuda_stream_view stream) const override
  {
    // lock_t lock(mtx);
    return {0, 0};
  }

  std::mutex mutable mtx;  // mutex for thread safe access to upstream
  cuda::stream_ordered_resource_view<cuda::memory_access::device>
    upstream_;  ///< The upstream resource used for satisfying allocation requests
};

}  // namespace rmm::mr
