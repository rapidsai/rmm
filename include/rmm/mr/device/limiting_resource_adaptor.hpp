/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <rmm/detail/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda/memory_resource>

#include <cstddef>

namespace rmm::mr {
/**
 * @brief Resource that uses `Upstream` to allocate memory and limits the total allocation.
 *
 * An instance of this resource can be constructed with an existing, upstream resource in order to
 * satisfy allocation requests, but any existing allocations will be untracked. Atomics are used to
 * make this thread-safe, but note that the `get_allocated_bytes` may not include in-flight
 * allocations.
 *
 * @tparam UpstreamPointer Type of the pointer to the upstream resource used for allocation.
 * @tparam Properties properties of the upstream resource (usually deduced with CTAD)
 */
template <typename UpstreamPointer, typename... Properties>
class limiting_resource_adaptor final : public device_memory_resource {
 public:
  using upstream_view_type = cuda::pmr::basic_resource_ptr<UpstreamPointer, Properties...>;
  /**
   * @brief Construct a new limiting resource adaptor using `upstream` to satisfy
   * allocation requests and limiting the total allocation amount possible.
   *
   * @throws `rmm::logic_error` if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory
   * @param allocation_limit Maximum memory allowed for this allocator.
   */
  limiting_resource_adaptor(upstream_view_type upstream,
                            std::size_t allocation_limit,
                            std::size_t alignment = rmm::detail::CUDA_ALLOCATION_ALIGNMENT)
    : allocation_limit_{allocation_limit},
      allocated_bytes_(0),
      alignment_(alignment),
      upstream_{upstream}
  {
    RMM_EXPECTS(upstream != upstream_view_type{nullptr},
                "Unexpected null upstream resource pointer.");
  }

  limiting_resource_adaptor()                                     = delete;
  ~limiting_resource_adaptor() override                           = default;
  limiting_resource_adaptor(limiting_resource_adaptor const&)     = delete;
  limiting_resource_adaptor(limiting_resource_adaptor&&) noexcept = delete;
  limiting_resource_adaptor& operator=(limiting_resource_adaptor const&) = delete;
  limiting_resource_adaptor& operator=(limiting_resource_adaptor&&) noexcept = delete;

  /**
   * @brief Return pointer to the upstream resource.
   *
   * @return View of the upstream resource.
   */
  [[nodiscard]] upstream_view_type get_upstream() const noexcept { return upstream_; }

  /**
   * @brief Checks whether the upstream resource supports streams.
   *
   * @return true The upstream resource supports streams
   * @return false The upstream resource does not support streams.
   */
  [[nodiscard]] bool supports_streams() const noexcept override { return true; }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   *
   * @return bool true if the upstream resource supports get_mem_info, false otherwise.
   */
  [[nodiscard]] bool supports_get_mem_info() const noexcept override { return false; }

  /**
   * @brief Query the number of bytes that have been allocated. Note that
   * this can not be used to know how large of an allocation is possible due
   * to both possible fragmentation and also internal page sizes and alignment
   * that is not tracked by this allocator.
   *
   * @return std::size_t number of bytes that have been allocated through this
   * allocator.
   */
  [[nodiscard]] std::size_t get_allocated_bytes() const { return allocated_bytes_; }

  /**
   * @brief Query the maximum number of bytes that this allocator is allowed
   * to allocate. This is the limit on the allocator and not a representation of
   * the underlying device. The device may not be able to support this limit.
   *
   * @return std::size_t max number of bytes allowed for this allocator
   */
  [[nodiscard]] std::size_t get_allocation_limit() const { return allocation_limit_; }

 private:
  /**
   * @brief Allocates memory of size at least `bytes` using the upstream
   * resource as long as it fits inside the allocation limit.
   *
   * The returned pointer has at least 256B alignment.
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
    auto const proposed_size = rmm::detail::align_up(bytes, alignment_);
    auto const old           = allocated_bytes_.fetch_add(proposed_size);
    if (old + proposed_size <= allocation_limit_) {
      try {
        return upstream_->allocate_async(bytes, stream.value());
      } catch (...) {
        allocated_bytes_ -= proposed_size;
        throw;
      }
    }

    allocated_bytes_ -= proposed_size;
    RMM_FAIL("Exceeded memory limit", rmm::out_of_memory);
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
    std::size_t allocated_size = rmm::detail::align_up(bytes, alignment_);
    upstream_->deallocate_async(ptr, bytes, stream.value());
    allocated_bytes_ -= allocated_size;
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
  [[nodiscard]] bool do_is_equal(
    cuda::pmr::memory_resource<memory_kind> const& other) const noexcept override
  {
    if (this == &other) { return true; }
    auto const* cast = dynamic_cast<limiting_resource_adaptor const*>(&other);
    if (cast != nullptr) { return upstream_ == cast->get_upstream(); }
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
  [[nodiscard]] std::pair<std::size_t, std::size_t> do_get_mem_info(
    cuda_stream_view stream) const override
  {
    return {allocation_limit_ - allocated_bytes_, allocation_limit_};
  }

  // maximum bytes this allocator is allowed to allocate.
  std::size_t allocation_limit_;

  // number of currently-allocated bytes
  std::atomic<std::size_t> allocated_bytes_;

  // todo: should be some way to ask the upstream...
  std::size_t alignment_;

  // The upstream resource used for satisfying allocation requests
  upstream_view_type upstream_;
};

}  // namespace rmm::mr
