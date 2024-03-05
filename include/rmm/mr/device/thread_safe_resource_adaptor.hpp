/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <mutex>

namespace rmm::mr {
/**
 * @addtogroup device_resource_adaptors
 * @{
 * @file
 */
/**
 * @brief Resource that adapts `Upstream` memory resource adaptor to be thread safe.
 *
 * An instance of this resource can be constructured with an existing, upstream resource in order
 * to satisfy allocation requests. This adaptor wraps allocations and deallocations from Upstream
 * in a mutex lock.
 *
 * @tparam Upstream Type of the upstream resource used for allocation/deallocation.
 */
template <typename Upstream>
class thread_safe_resource_adaptor final : public device_memory_resource {
 public:
  using lock_t = std::lock_guard<std::mutex>;  ///< Type of lock used to synchronize access

  /**
   * @brief Construct a new thread safe resource adaptor using `upstream` to satisfy
   * allocation requests.
   *
   * All allocations and frees are protected by a mutex lock
   *
   * @throws rmm::logic_error if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory.
   */
  thread_safe_resource_adaptor(Upstream* upstream) : upstream_{upstream}
  {
    RMM_EXPECTS(nullptr != upstream, "Unexpected null upstream resource pointer.");
  }

  thread_safe_resource_adaptor()                                               = delete;
  ~thread_safe_resource_adaptor() override                                     = default;
  thread_safe_resource_adaptor(thread_safe_resource_adaptor const&)            = delete;
  thread_safe_resource_adaptor(thread_safe_resource_adaptor&&)                 = delete;
  thread_safe_resource_adaptor& operator=(thread_safe_resource_adaptor const&) = delete;
  thread_safe_resource_adaptor& operator=(thread_safe_resource_adaptor&&)      = delete;

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept
  {
    return upstream_;
  }

  /**
   * @briefreturn{Upstream* to the upstream memory resource}
   */
  [[nodiscard]] Upstream* get_upstream() const noexcept { return upstream_; }

 private:
  /**
   * @brief Allocates memory of size at least `bytes` using the upstream
   * resource with thread safety.
   *
   * @throws rmm::bad_alloc if the requested allocation could not be fulfilled
   * by the upstream resource.
   *
   * @param bytes The size, in bytes, of the allocation
   * @param stream Stream on which to perform the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
  {
    lock_t lock(mtx);
    return upstream_->allocate(bytes, stream);
  }

  /**
   * @brief Free allocation of size `bytes` pointed to to by `ptr`.s
   *
   * @param ptr Pointer to be deallocated
   * @param bytes Size of the allocation
   * @param stream Stream on which to perform the deallocation
   */
  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) override
  {
    lock_t lock(mtx);
    upstream_->deallocate(ptr, bytes, stream);
  }

  /**
   * @brief Compare the upstream resource to another.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equivalent
   */
  bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    if (this == &other) { return true; }
    auto cast = dynamic_cast<thread_safe_resource_adaptor<Upstream> const*>(&other);
    if (cast == nullptr) { return upstream_->is_equal(other); }
    return get_upstream_resource() == cast->get_upstream_resource();
  }

  std::mutex mutable mtx;  // mutex for thread safe access to upstream
  Upstream* upstream_;     ///< The upstream resource used for satisfying allocation requests
};

/** @} */  // end of group
}  // namespace rmm::mr
