/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <rmm/prefetch.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <mutex>
#include <shared_mutex>
#include <stack>

namespace rmm::mr {
/**
 * @addtogroup device_resource_adaptors
 * @{
 * @file
 */
/**
 * @brief Resource that prefetches all memory allocations.
 *
 * @tparam Upstream Type of the upstream resource used for
 * allocation/deallocation.
 */
template <typename Upstream>
class prefetch_resource_adaptor final : public device_memory_resource {
 public:
  /**
   * @brief Construct a new prefetch resource adaptor using `upstream` to satisfy
   * allocation requests.
   *
   * @throws rmm::logic_error if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory
   */
  prefetch_resource_adaptor(Upstream* upstream) : upstream_{upstream}
  {
    RMM_EXPECTS(nullptr != upstream, "Unexpected null upstream resource pointer.");
  }

  prefetch_resource_adaptor()                                            = delete;
  ~prefetch_resource_adaptor() override                                  = default;
  prefetch_resource_adaptor(prefetch_resource_adaptor const&)            = delete;
  prefetch_resource_adaptor& operator=(prefetch_resource_adaptor const&) = delete;
  prefetch_resource_adaptor(prefetch_resource_adaptor&&) noexcept =
    default;  ///< @default_move_constructor
  prefetch_resource_adaptor& operator=(prefetch_resource_adaptor&&) noexcept =
    default;  ///< @default_move_assignment{prefetch_resource_adaptor}

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
   * resource as long as it fits inside the allocation limit.
   *
   * @note The allocation is always prefetched to the current device.
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
    void* ptr = upstream_->allocate(bytes, stream);
    rmm::prefetch(ptr, bytes, rmm::get_current_cuda_device(), stream);
    return ptr;
  }

  /**
   * @brief Free allocation of size `bytes` pointed to by `ptr`
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
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    if (this == &other) { return true; }
    auto cast = dynamic_cast<prefetch_resource_adaptor<Upstream> const*>(&other);
    if (cast == nullptr) { return upstream_->is_equal(other); }
    return get_upstream_resource() == cast->get_upstream_resource();
  }

  Upstream* upstream_;  // the upstream resource used for satisfying allocation requests
};

/** @} */  // end of group
}  // namespace rmm::mr
