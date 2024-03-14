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

#include <rmm/aligned.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/fixed_size_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <map>
#include <memory>
#include <vector>

namespace rmm::mr {
/**
 * @addtogroup device_memory_resources
 * @{
 * @file
 */

/**
 * @brief Allocates memory from upstream resources associated with bin sizes.
 *
 * @tparam UpstreamResource memory_resource to use for allocations that don't fall within any
 * configured bin size. Implements rmm::mr::device_memory_resource interface.
 */
template <typename Upstream>
class binning_memory_resource final : public device_memory_resource {
 public:
  /**
   * @brief Construct a new binning memory resource object.
   *
   * Initially has no bins, so simply uses the upstream_resource until bin resources are added
   * with `add_bin`.
   *
   * @throws rmm::logic_error if size_base is not a power of two.
   *
   * @param upstream_resource The upstream memory resource used to allocate bin pools.
   */
  explicit binning_memory_resource(Upstream* upstream_resource)
    : upstream_mr_{[upstream_resource]() {
        RMM_EXPECTS(nullptr != upstream_resource, "Unexpected null upstream pointer.");
        return upstream_resource;
      }()}
  {
  }

  /**
   * @brief Construct a new binning memory resource object with a range of initial bins.
   *
   * Constructs a new binning memory resource and adds bins backed by `fixed_size_memory_resource`
   * in the range [2^min_size_exponent, 2^max_size_exponent]. For example if `min_size_exponent==18`
   * and `max_size_exponent==22`, creates bins of sizes 256KiB, 512KiB, 1024KiB, 2048KiB and
   * 4096KiB.
   *
   * @param upstream_resource The upstream memory resource used to allocate bin pools.
   * @param min_size_exponent The minimum base-2 exponent bin size.
   * @param max_size_exponent The maximum base-2 exponent bin size.
   */
  binning_memory_resource(Upstream* upstream_resource,
                          int8_t min_size_exponent,  // NOLINT(bugprone-easily-swappable-parameters)
                          int8_t max_size_exponent)
    : upstream_mr_{[upstream_resource]() {
        RMM_EXPECTS(nullptr != upstream_resource, "Unexpected null upstream pointer.");
        return upstream_resource;
      }()}
  {
    for (auto i = min_size_exponent; i <= max_size_exponent; i++) {
      add_bin(1 << i);
    }
  }

  /**
   * @brief Destroy the binning_memory_resource and free all memory allocated from the upstream
   * resource.
   */
  ~binning_memory_resource() override = default;

  binning_memory_resource()                                          = delete;
  binning_memory_resource(binning_memory_resource const&)            = delete;
  binning_memory_resource(binning_memory_resource&&)                 = delete;
  binning_memory_resource& operator=(binning_memory_resource const&) = delete;
  binning_memory_resource& operator=(binning_memory_resource&&)      = delete;

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept
  {
    return upstream_mr_;
  }

  /**
   * @briefreturn{Upstream* to the upstream memory resource}
   */
  [[nodiscard]] Upstream* get_upstream() const noexcept { return upstream_mr_; }

  /**
   * @brief Add a bin allocator to this resource
   *
   * Adds `bin_resource` if it is not null; otherwise constructs and adds a
   * fixed_size_memory_resource.
   *
   * This bin will be used for any allocation smaller than `allocation_size` that is larger than
   * the next smaller bin's allocation size.
   *
   * If there is already a bin of the specified size nothing is changed.
   *
   * This function is not thread safe.
   *
   * @param allocation_size The maximum size that this bin allocates
   * @param bin_resource The memory resource for the bin
   */
  void add_bin(std::size_t allocation_size, device_memory_resource* bin_resource = nullptr)
  {
    allocation_size = rmm::align_up(allocation_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

    if (nullptr != bin_resource) {
      resource_bins_.insert({allocation_size, bin_resource});
    } else if (resource_bins_.count(allocation_size) == 0) {  // do nothing if bin already exists

      owned_bin_resources_.push_back(
        std::make_unique<fixed_size_memory_resource<Upstream>>(upstream_mr_, allocation_size));
      resource_bins_.insert({allocation_size, owned_bin_resources_.back().get()});
    }
  }

 private:
  /**
   * @brief Get the memory resource for the requested size
   *
   * Chooses a memory_resource that allocates the smallest blocks at least as large as `bytes`.
   *
   * @param bytes Requested allocation size in bytes
   * @return Get the resource reference for the requested size.
   */
  rmm::device_async_resource_ref get_resource_ref(std::size_t bytes)
  {
    auto iter = resource_bins_.lower_bound(bytes);
    return (iter != resource_bins_.cend()) ? rmm::device_async_resource_ref{iter->second}
                                           : get_upstream_resource();
  }

  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
  {
    if (bytes <= 0) { return nullptr; }
    return get_resource_ref(bytes).allocate_async(bytes, stream);
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   */
  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) override
  {
    get_resource_ref(bytes).deallocate_async(ptr, bytes, stream);
  }

  Upstream* upstream_mr_;  // The upstream memory_resource from which to allocate blocks.

  std::vector<std::unique_ptr<fixed_size_memory_resource<Upstream>>> owned_bin_resources_;

  std::map<std::size_t, device_memory_resource*> resource_bins_;
};

/** @} */  // end of group
}  // namespace rmm::mr
