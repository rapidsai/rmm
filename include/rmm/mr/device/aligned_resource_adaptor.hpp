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

#include <optional>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace rmm::mr {
/**
 * @brief Resource that adapts `Upstream` memory resource to allocate memory in a specified
 * alignment size.
 *
 * An instance of this resource can be constructed with an existing, upstream resource in order
 * to satisfy allocation requests. This adaptor wraps allocations and deallocations from Upstream
 * using the given alignment size.
 *
 * By default, any address returned by one of the memory allocation routines from the CUDA driver or
 * runtime API is always aligned to at least 256 bytes. For some use cases, such as GPUDirect
 * Storage (GDS), allocations need to be aligned to a larger size (4 KiB for GDS) in order to avoid
 * additional copies to bounce buffers.
 *
 * Since a larger alignment size has some additional overhead, the user can specify a threshold
 * size. If an allocation's size falls below the threshold, it is aligned to the default size. Only
 * allocations with a size above the threshold are aligned to the custom alignment size.
 *
 * @tparam Upstream Type of the upstream resource used for allocation/deallocation.
 */
template <typename Upstream>
class aligned_resource_adaptor final : public device_memory_resource {
 public:
  /**
   * @brief Construct an aligned resource adaptor using `upstream` to satisfy allocation requests.
   *
   * @throws `rmm::logic_error` if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory.
   * @param allocation_alignment The size used for allocation alignment.
   * @param alignment_threshold Only allocations with a size larger than or equal to this threshold
   * are aligned.
   */
  explicit aligned_resource_adaptor(Upstream* upstream,
                                    std::size_t allocation_alignment = default_allocation_alignment,
                                    std::size_t alignment_threshold  = default_alignment_threshold)
    : upstream_{upstream},
      allocation_alignment_{allocation_alignment},
      alignment_threshold_{alignment_threshold}
  {
    RMM_EXPECTS(nullptr != upstream, "Unexpected null upstream resource pointer.");
    RMM_EXPECTS(allocation_alignment % 256 == 0, "Allocation alignment is not a multiple of 256.");
    RMM_EXPECTS(alignment_threshold % 256 == 0, "Alignment threshold is not a multiple of 256.");
  }

  aligned_resource_adaptor()                                = delete;
  ~aligned_resource_adaptor() override                      = default;
  aligned_resource_adaptor(aligned_resource_adaptor const&) = delete;
  aligned_resource_adaptor(aligned_resource_adaptor&&)      = delete;
  aligned_resource_adaptor& operator=(aligned_resource_adaptor const&) = delete;
  aligned_resource_adaptor& operator=(aligned_resource_adaptor&&) = delete;

  /**
   * @brief Get the upstream memory resource.
   *
   * @return Upstream* pointer to a memory resource object.
   */
  Upstream* get_upstream() const noexcept { return upstream_; }

  /**
   * @copydoc rmm::mr::device_memory_resource::supports_streams()
   */
  [[nodiscard]] bool supports_streams() const noexcept override
  {
    return upstream_->supports_streams();
  }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   *
   * @return bool true if the upstream resource supports get_mem_info, false otherwise.
   */
  [[nodiscard]] bool supports_get_mem_info() const noexcept override
  {
    return upstream_->supports_get_mem_info();
  }

 private:
  static constexpr std::size_t default_allocation_alignment = 256;
  static constexpr std::size_t default_alignment_threshold  = 0;

  /**
   * @brief Allocates memory of size at least `bytes` using the upstream resource with the specified
   * alignment.
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
    if (allocation_alignment_ == default_allocation_alignment || bytes < alignment_threshold_) {
      return upstream_->allocate(bytes, stream);
    } else {
      auto const aligned_size = rmm::detail::align_up(bytes, allocation_alignment_);
      auto const size         = aligned_size + allocation_alignment_ - default_allocation_alignment;
      auto const address      = reinterpret_cast<std::size_t>(upstream_->allocate(size, stream));
      auto const aligned_address = rmm::detail::align_up(address, allocation_alignment_);
      auto const head_size       = aligned_address - address;
      auto const tail_size       = size - head_size - aligned_size;
      if (head_size != 0) { upstream_->deallocate(reinterpret_cast<void*>(address), head_size); }
      if (tail_size != 0) {
        upstream_->deallocate(reinterpret_cast<void*>(aligned_address + aligned_size), tail_size);
      }
      return reinterpret_cast<void*>(aligned_address);
    }
  }

  /**
   * @brief Free allocation of size `bytes` pointed to to by `p` and log the deallocation.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   * @param bytes Size of the allocation
   * @param stream Stream on which to perform the deallocation
   */
  void do_deallocate(void* p, std::size_t bytes, cuda_stream_view stream) override
  {
    if (allocation_alignment_ == default_allocation_alignment || bytes < alignment_threshold_) {
      upstream_->deallocate(p, bytes, stream);
    } else {
      upstream_->deallocate(p, rmm::detail::align_up(bytes, allocation_alignment_), stream);
    }
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
  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    if (this == &other)
      return true;
    else {
      auto aligned_other = dynamic_cast<aligned_resource_adaptor<Upstream> const*>(&other);
      if (aligned_other != nullptr)
        return upstream_->is_equal(*aligned_other->get_upstream());
      else
        return upstream_->is_equal(other);
    }
  }

  /**
   * @brief Get free and available memory from upstream resource.
   *
   * @throws `rmm::cuda_error` if unable to retrieve memory info.
   *
   * @param stream Stream on which to get the mem info.
   * @return std::pair containing free_size and total_size of memory
   */
  [[nodiscard]] std::pair<size_t, size_t> do_get_mem_info(cuda_stream_view stream) const override
  {
    return upstream_->get_mem_info(stream);
  }

  Upstream* upstream_;  ///< The upstream resource used for satisfying allocation requests
  std::size_t allocation_alignment_;  ///< The size used for allocation alignment
  std::size_t alignment_threshold_;   ///< The size above which allocations should be aligned
};

}  // namespace rmm::mr
