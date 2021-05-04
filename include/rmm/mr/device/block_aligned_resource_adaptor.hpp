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

namespace rmm {
namespace mr {
/**
 * @brief Resource that adapts `Upstream` memory resource to allocate memory in multiples of
 * filesystem block size (default to 4096).
 *
 * An instance of this resource can be constructed with an existing, upstream resource in order
 * to satisfy allocation requests. This adaptor wraps allocations and deallocations from Upstream
 * with block size alignment.
 *
 * @tparam Upstream Type of the upstream resource used for allocation/deallocation.
 */
template <typename Upstream>
class block_aligned_resource_adaptor final : public device_memory_resource {
 public:
  /**
   * @brief Construct a new block size aligned resource adaptor using `upstream` to satisfy
   * allocation requests.
   *
   * If the allocation size is smaller or equal to the block size, the default alignment from
   * `upstream` is used; if the allocation size is larger than the block size, it is aligned up to
   * the block size.
   *
   * @throws `rmm::logic_error` if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory.
   */
  block_aligned_resource_adaptor(Upstream* upstream,
                                 std::optional<std::size_t> block_size = std::nullopt)
    : upstream_{upstream}, block_size_{block_size.value_or(default_block_size)}
  {
    RMM_EXPECTS(nullptr != upstream, "Unexpected null upstream resource pointer.");
  }

  block_aligned_resource_adaptor()                                      = delete;
  ~block_aligned_resource_adaptor() override                            = default;
  block_aligned_resource_adaptor(block_aligned_resource_adaptor const&) = delete;
  block_aligned_resource_adaptor(block_aligned_resource_adaptor&&)      = delete;
  block_aligned_resource_adaptor& operator=(block_aligned_resource_adaptor const&) = delete;
  block_aligned_resource_adaptor& operator=(block_aligned_resource_adaptor&&) = delete;

  /**
   * @brief Get the upstream memory resource.
   *
   * @return Upstream* pointer to a memory resource object.
   */
  Upstream* get_upstream() const noexcept { return upstream_; }

  /**
   * @copydoc rmm::mr::device_memory_resource::supports_streams()
   */
  bool supports_streams() const noexcept override { return upstream_->supports_streams(); }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   *
   * @return bool true if the upstream resource supports get_mem_info, false otherwise.
   */
  bool supports_get_mem_info() const noexcept override { return upstream_->supports_streams(); }

 private:
  static constexpr std::size_t default_block_size = 4096;

  /**
   * @brief Allocates memory of size at least `bytes` using the upstream
   * resource with block size alignment.
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
    return upstream_->allocate(block_align(bytes), stream);
  }

  /**
   * @brief Free allocation of size `bytes` pointed to to by `p` and log the
   * deallocation.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   * @param bytes Size of the allocation
   * @param stream Stream on which to perform the deallocation
   */
  void do_deallocate(void* p, std::size_t bytes, cuda_stream_view stream) override
  {
    upstream_->deallocate(p, block_align(bytes), stream);
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
  bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    if (this == &other)
      return true;
    else {
      auto block_aligned_other =
        dynamic_cast<block_aligned_resource_adaptor<Upstream> const*>(&other);
      if (block_aligned_other != nullptr)
        return upstream_->is_equal(*block_aligned_other->get_upstream());
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
  std::pair<size_t, size_t> do_get_mem_info(cuda_stream_view stream) const override
  {
    return upstream_->get_mem_info(stream);
  }

  /**
   * @brief Align up to nearest multiple of the block size.
   *
   * @param[in] bytes value to align
   *
   * @return Return the aligned value
   */
  std::size_t block_align(std::size_t bytes)
  {
    return bytes <= block_size_ ? bytes : rmm::detail::align_up(bytes, block_size_);
  }

  Upstream* upstream_;      ///< The upstream resource used for satisfying allocation requests
  std::size_t block_size_;  ///< The block size used for alignment
};

}  // namespace mr
}  // namespace rmm
