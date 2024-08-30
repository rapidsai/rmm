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

#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <mutex>
#include <unordered_set>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup device_resource_adaptors
 * @{
 * @file
 */

template <typename Upstream, typename ExceptionType = rmm::out_of_memory>
class failure_alternate_resource_adaptor final : public device_memory_resource {
 public:
  using exception_type = ExceptionType;  ///< The type of exception this object catches/throws

  /**
   * @brief Construct a new `failure_alternate_resource_adaptor` using `upstream` to satisfy
   * allocation requests.
   *
   * @throws rmm::logic_error if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory
   * @param alternate_upstream  The resource used for alternate allocating/deallocating device
   * memory
   */
  failure_alternate_resource_adaptor(Upstream* upstream, Upstream* alternate_upstream)
    : upstream_{upstream}, alternate_upstream_{alternate_upstream}
  {
    RMM_EXPECTS(nullptr != upstream, "Unexpected null upstream resource pointer.");
    RMM_EXPECTS(nullptr != alternate_upstream,
                "Unexpected null alternate upstream resource pointer.");
  }

  failure_alternate_resource_adaptor()                                          = delete;
  ~failure_alternate_resource_adaptor() override                                = default;
  failure_alternate_resource_adaptor(failure_alternate_resource_adaptor const&) = delete;
  failure_alternate_resource_adaptor& operator=(failure_alternate_resource_adaptor const&) = delete;
  failure_alternate_resource_adaptor(failure_alternate_resource_adaptor&&) noexcept =
    default;  ///< @default_move_constructor
  failure_alternate_resource_adaptor& operator=(failure_alternate_resource_adaptor&&) noexcept =
    default;  ///< @default_move_assignment{failure_alternate_resource_adaptor}

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
  using lock_guard = std::lock_guard<std::mutex>;

  /**
   * @brief Allocates memory of size at least `bytes` using the upstream
   * resource.
   *
   * @throws `exception_type` if the requested allocation could not be fulfilled
   * by the upstream resource.
   *
   * @param bytes The size, in bytes, of the allocation
   * @param stream Stream on which to perform the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
  {
    void* ret{};
    try {
      ret = upstream_->allocate(bytes, stream);
    } catch (exception_type const& e) {
      ret = alternate_upstream_->allocate(bytes, stream);
      lock_guard lock(mtx_);
      alternate_allocations_.insert(ret);
    }
    return ret;
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
    std::size_t count{0};
    {
      lock_guard lock(mtx_);
      count = alternate_allocations_.erase(ptr);
    }
    if (count > 0) {
      alternate_upstream_->deallocate(ptr, bytes, stream);
    } else {
      upstream_->deallocate(ptr, bytes, stream);
    }
  }

  /**
   * @brief Compare the upstream resource to another.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    if (this == &other) { return true; }
    auto cast = dynamic_cast<failure_alternate_resource_adaptor<Upstream> const*>(&other);
    if (cast == nullptr) { return upstream_->is_equal(other); }
    return get_upstream_resource() == cast->get_upstream_resource();
  }

  Upstream* upstream_;            // the upstream used for satisfying allocation requests
  Upstream* alternate_upstream_;  // the upstream used for satisfying alternate allocation requests
  std::unordered_set<void*> alternate_allocations_;  // set of alternate allocations
  mutable std::mutex mtx_;                           // Mutex for exclusive lock.
};

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
