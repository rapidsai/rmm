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

#include <rmm/detail/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cstddef>

namespace rmm::mr {

/**
 * @brief Callback function type used by oom_callback_resource_adaptor
 *
 */
using oom_callback_t = bool (*)(std::size_t, void*);

/**
 * @brief Resource that uses `Upstream` to allocate memory and calls `callback`
 * when allocations throws `std::bad_alloc`.
 *
 * An instance of this resource can be constructed with an existing, upstream
 * resource in order to satisfy allocation requests.
 *
 * The callback function takes an allocation size and a closure and returns
 * whether to retry the allocation or throw `std::bad_alloc`.
 *
 * @tparam Upstream Type of the upstream resource used for
 * allocation/deallocation.
 */
template <typename Upstream>
class oom_callback_resource_adaptor final : public device_memory_resource {
 public:
  /**
   * @brief Construct a new OOM callback resource adaptor using `upstream` to satisfy
   * allocation requests.
   *
   * @throws `rmm::logic_error` if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory
   * @param callback Callback function
   * @param closure Extra argument passed to `callback`
   */
  oom_callback_resource_adaptor(Upstream* upstream, oom_callback_t callback, void* closure)
    : upstream_{upstream}, callback_{callback}, closure_{closure}
  {
    RMM_EXPECTS(nullptr != upstream, "Unexpected null upstream resource pointer.");
  }

  oom_callback_resource_adaptor()                                     = delete;
  ~oom_callback_resource_adaptor() override                           = default;
  oom_callback_resource_adaptor(oom_callback_resource_adaptor const&) = delete;
  oom_callback_resource_adaptor& operator=(oom_callback_resource_adaptor const&) = delete;
  oom_callback_resource_adaptor(oom_callback_resource_adaptor&&) noexcept        = default;
  oom_callback_resource_adaptor& operator=(oom_callback_resource_adaptor&&) noexcept = default;

  /**
   * @brief Return pointer to the upstream resource.
   *
   * @return Upstream* Pointer to the upstream resource.
   */
  Upstream* get_upstream() const noexcept { return upstream_; }

  /**
   * @brief Checks whether the upstream resource supports streams.
   *
   * @return true The upstream resource supports streams
   * @return false The upstream resource does not support streams.
   */
  bool supports_streams() const noexcept override { return upstream_->supports_streams(); }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   *
   * @return bool true if the upstream resource supports get_mem_info, false otherwise.
   */
  bool supports_get_mem_info() const noexcept override
  {
    return upstream_->supports_get_mem_info();
  }

 private:
  /**
   * @brief Allocates memory of size at least `bytes` using the upstream
   * resource.
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
    void* ret;

    while (true) {
      try {
        ret = upstream_->allocate(bytes, stream);
        break;
      } catch (std::bad_alloc const& e) {
        // Call callback
        if (!(*callback_)(bytes, closure_)) { throw; }
      }
    }
    return ret;
  }

  /**
   * @brief Free allocation of size `bytes` pointed to by `ptr`
   *
   * @throws Nothing.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes Size of the allocation
   * @param stream Stream on which to perform the deallocation
   */
  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) override
  {
    upstream_->deallocate(ptr, bytes, stream);
  }

  /**
   * @brief Compare the upstream resource to another.
   *
   * @throws Nothing.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    if (this == &other) { return true; }
    auto cast = dynamic_cast<oom_callback_resource_adaptor<Upstream> const*>(&other);
    return cast != nullptr ? upstream_->is_equal(*cast->get_upstream())
                           : upstream_->is_equal(other);
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
    return upstream_->get_mem_info(stream);
  }

  Upstream* upstream_;  // the upstream resource used for satisfying allocation requests
  oom_callback_t callback_;
  void* closure_;
};

}  // namespace rmm::mr
