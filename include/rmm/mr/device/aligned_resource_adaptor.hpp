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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cstddef>
#include <mutex>
#include <optional>
#include <unordered_map>

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
   * @throws `rmm::logic_error` if `allocation_alignment` is not a power of 2
   *
   * @param upstream The resource used for allocating/deallocating device memory.
   * @param alignment The size used for allocation alignment.
   * @param alignment_threshold Only allocations with a size larger than or equal to this threshold
   * are aligned.
   */
  explicit aligned_resource_adaptor(Upstream* upstream,
                                    std::size_t alignment = rmm::detail::CUDA_ALLOCATION_ALIGNMENT,
                                    std::size_t alignment_threshold = default_alignment_threshold)
    : upstream_{upstream}, alignment_{alignment}, alignment_threshold_{alignment_threshold}
  {
    RMM_EXPECTS(nullptr != upstream, "Unexpected null upstream resource pointer.");
    RMM_EXPECTS(rmm::detail::is_supported_alignment(alignment),
                "Allocation alignment is not a power of 2.");
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
  static constexpr std::size_t default_alignment_threshold = 0;
  using lock_guard                                         = std::lock_guard<std::mutex>;

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
    if (alignment_ == rmm::detail::CUDA_ALLOCATION_ALIGNMENT || bytes < alignment_threshold_) {
      return upstream_->allocate(bytes, stream);
    }
    auto const size = upstream_allocation_size(bytes);
    void* pointer   = upstream_->allocate(size, stream);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto const address         = reinterpret_cast<std::size_t>(pointer);
    auto const aligned_address = rmm::detail::align_up(address, alignment_);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
    void* aligned_pointer = reinterpret_cast<void*>(aligned_address);
    if (pointer != aligned_pointer) {
      lock_guard lock(mtx_);
      pointers_.emplace(aligned_pointer, pointer);
    }
    return aligned_pointer;
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
  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) override
  {
    if (alignment_ == rmm::detail::CUDA_ALLOCATION_ALIGNMENT || bytes < alignment_threshold_) {
      upstream_->deallocate(ptr, bytes, stream);
    } else {
      {
        lock_guard lock(mtx_);
        auto const iter = pointers_.find(ptr);
        if (iter != pointers_.end()) {
          ptr = iter->second;
          pointers_.erase(iter);
        }
      }
      upstream_->deallocate(ptr, upstream_allocation_size(bytes), stream);
    }
  }

  /**
   * @brief Compare this resource to another.
   *
   * @throws Nothing.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equivalent
   */
  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    if (this == &other) { return true; }
    auto cast = dynamic_cast<aligned_resource_adaptor<Upstream> const*>(&other);
    return cast != nullptr && upstream_->is_equal(*cast->get_upstream()) &&
           alignment_ == cast->alignment_ && alignment_threshold_ == cast->alignment_threshold_;
  }

  /**
   * @brief Get free and available memory from upstream resource.
   *
   * The free size may not be fully allocatable because of alignment requirements.
   *
   * @throws `rmm::cuda_error` if unable to retrieve memory info.
   *
   * @param stream Stream on which to get the mem info.
   * @return std::pair containing free_size and total_size of memory
   */
  [[nodiscard]] std::pair<std::size_t, std::size_t> do_get_mem_info(
    cuda_stream_view stream) const override
  {
    return upstream_->get_mem_info(stream);
  }

  /**
   * @brief Calculate the allocation size needed from upstream to account for alignments of both the
   * size and the base pointer.
   *
   * @param bytes The requested allocation size.
   * @return Allocation size needed from upstream to align both the size and the base pointer.
   */
  std::size_t upstream_allocation_size(std::size_t bytes) const
  {
    auto const aligned_size = rmm::detail::align_up(bytes, alignment_);
    return aligned_size + alignment_ - rmm::detail::CUDA_ALLOCATION_ALIGNMENT;
  }

  Upstream* upstream_;  ///< The upstream resource used for satisfying allocation requests
  std::unordered_map<void*, void*> pointers_;  ///< Map of aligned pointers to upstream pointers.
  std::size_t alignment_;                      ///< The size used for allocation alignment
  std::size_t alignment_threshold_;  ///< The size above which allocations should be aligned
  mutable std::mutex mtx_;           ///< Mutex for exclusive lock.
};

}  // namespace rmm::mr
