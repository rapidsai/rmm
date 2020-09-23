/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <execinfo.h>
#include <map>
#include <mutex>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <sstream>

namespace rmm {
namespace mr {
/**
 * @brief Resource that uses `Upstream` to allocate memory and tracks allocations.
 *
 * An instance of this resource can be constructed with an existing, upstream
 * resource in order to satisfy allocation requests, but any existing allocations
 * will be untracked. Tracking data is heavy as we store a stack frame, size and pointer
 * for each allocation. This is intended as a debug adaptor and shouldn't be used in
 * performance sensitive code.
 *
 * @tparam Upstream Type of the upstream resource used for
 * allocation/deallocation.
 */
template <typename Upstream>
class tracking_resource_adaptor final : public device_memory_resource {
 public:
  using lock_t = std::lock_guard<std::mutex>;

  /**
   * @brief Construct a new tracking resource adaptor using `upstream` to satisfy
   * allocation requests and tracking active allocations.
   *
   * @throws `rmm::logic_error` if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory
   */
  tracking_resource_adaptor(Upstream* upstream) : upstream_{upstream}
  {
    RMM_EXPECTS(nullptr != upstream, "Unexpected null upstream resource pointer.");
  }

  tracking_resource_adaptor()                                 = delete;
  ~tracking_resource_adaptor()                                = default;
  tracking_resource_adaptor(tracking_resource_adaptor const&) = delete;
  tracking_resource_adaptor(tracking_resource_adaptor&&)      = default;
  tracking_resource_adaptor& operator=(tracking_resource_adaptor const&) = delete;
  tracking_resource_adaptor& operator=(tracking_resource_adaptor&&) = default;

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

  /**
   * @brief Print any outstanding allocations to debug log
   *
   */
  void print_outstanding_allocations() const
  {
    lock_t lock(mtx);
    std::ostringstream oss;
    for (auto const& al : allocations) {
      oss << std::endl
          << "Allocation " << al.first << " of " << al.second.allocation_size
          << "bytes with callstack:" << std::endl;
      std::unique_ptr<char*, decltype(&::free)> strings(
        backtrace_symbols(al.second.stack_ptrs.data(), al.second.stack_ptrs.size()), &::free);
      if (strings.get() == nullptr) {
        oss << "But no stack trace could be found!" << std::endl;
      } else {
        ///@todo: support for demangling of C++ symbol names
        for (int i = 0; i < al.second.stack_ptrs.size(); ++i) {
          oss << "#" << i << " in " << strings.get()[i] << std::endl;
        }
      }
    }

    RMM_LOG_DEBUG("Outstanding Allocations: {}", oss.str());
  }

  /**
   * @brief Get the number of outstanding allocations
   *
   * @return std::size_t number of allocations still outstanding
   */
  std::size_t get_num_outstanding_allocations() const { return allocations.size(); };

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
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override
  {
    void* p = [&]() {
      try {
        return upstream_->allocate(bytes, stream);
      } catch (std::exception const& e) {
        RMM_LOG_ERROR("[A][Stream {}][Upstream {}B][FAILURE {}]",
                      reinterpret_cast<void*>(stream),
                      bytes,
                      e.what());
        throw;
      }
    }();

    // track it.
    {
      lock_t lock(mtx);
      allocations.emplace(p, bytes);
    }

    return p;
  }

  /**
   * @brief Free allocation of size `bytes` pointed to by `p`
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   * @param bytes Size of the allocation
   * @param stream Stream on which to perform the deallocation
   */
  void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override
  {
    {
      lock_t lock(mtx);
      allocations.erase(p);
    }
    upstream_->deallocate(p, bytes, stream);
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
    if (this == &other)
      return true;
    else {
      tracking_resource_adaptor<Upstream> const* cast =
        dynamic_cast<tracking_resource_adaptor<Upstream> const*>(&other);
      if (cast != nullptr)
        return upstream_->is_equal(*cast->get_upstream());
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
   * @return std::pair contaiing free_size and total_size of memory
   */
  std::pair<std::size_t, std::size_t> do_get_mem_info(cudaStream_t stream) const override
  {
    return upstream_->get_mem_info(stream);
  }

  struct allocation_info {
    std::vector<void*> stack_ptrs;
    std::size_t allocation_size;

    allocation_info() : allocation_size(0){};
    allocation_info(std::size_t size) : allocation_size(size)
    {
      // store off a stack for this allocation
      const int MaxStackDepth = 64;
      void* stack[MaxStackDepth];
      auto depth = backtrace(stack, MaxStackDepth);
      stack_ptrs.insert(stack_ptrs.end(), &stack[0], &stack[depth]);
    };
  };

  // map of active allocations
  std::map<void*, allocation_info> allocations;

  std::mutex mutable mtx;  // mutex for thread safe access to allocations

  Upstream* upstream_;  ///< The upstream resource used for satisfying
                        ///< allocation requests
};

/**
 * @brief Convenience factory to return a `tracking_resource_adaptor` around the
 * upstream resource `upstream`.
 *
 * @tparam Upstream Type of the upstream `device_memory_resource`.
 * @param upstream Pointer to the upstream resource
 */
template <typename Upstream>
tracking_resource_adaptor<Upstream> make_tracking_adaptor(Upstream* upstream)
{
  return tracking_resource_adaptor<Upstream>{upstream};
}

}  // namespace mr
}  // namespace rmm
