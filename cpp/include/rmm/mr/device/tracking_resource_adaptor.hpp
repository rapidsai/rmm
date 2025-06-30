/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
#include <rmm/detail/stack_trace.hpp>
#include <rmm/logger.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <atomic>
#include <cstddef>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <sstream>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup device_resource_adaptors
 * @{
 * @file
 */
/**
 * @brief Resource that uses `Upstream` to allocate memory and tracks allocations.
 *
 * An instance of this resource can be constructed with an existing, upstream
 * resource in order to satisfy allocation requests, but any existing allocations
 * will be untracked. Tracking stores a size and pointer for every allocation, and a stack
 * frame if `capture_stacks` is true, so it can add significant overhead.
 * `tracking_resource_adaptor` is intended as a debug adaptor and shouldn't be used in
 * performance-sensitive code. Note that callstacks may not contain all symbols unless
 * the project is linked with `-rdynamic`. This can be accomplished with
 * `add_link_options(-rdynamic)` in cmake.
 *
 * @tparam Upstream Type of the upstream resource used for
 * allocation/deallocation.
 */
template <typename Upstream>
class tracking_resource_adaptor final : public device_memory_resource {
 public:
  using read_lock_t =
    std::shared_lock<std::shared_mutex>;  ///< Type of lock used to synchronize read access
  using write_lock_t =
    std::unique_lock<std::shared_mutex>;  ///< Type of lock used to synchronize write access
  /**
   * @brief Information stored about an allocation. Includes the size
   * and a stack trace if the `tracking_resource_adaptor` was initialized
   * to capture stacks.
   *
   */
  struct allocation_info {
    std::unique_ptr<rmm::detail::stack_trace> strace;  ///< Stack trace of the allocation
    std::size_t allocation_size;                       ///< Size of the allocation

    allocation_info() = delete;
    /**
     * @brief Construct a new allocation info object
     *
     * @param size Size of the allocation
     * @param capture_stack If true, capture the stack trace for the allocation
     */
    allocation_info(std::size_t size, bool capture_stack)
      : strace{[&]() {
          return capture_stack ? std::make_unique<rmm::detail::stack_trace>() : nullptr;
        }()},
        allocation_size{size} {};
  };

  /**
   * @brief Construct a new tracking resource adaptor using `upstream` to satisfy
   * allocation requests.
   *
   * @param upstream The resource used for allocating/deallocating device memory
   * @param capture_stacks If true, capture stacks for allocation calls
   */
  tracking_resource_adaptor(device_async_resource_ref upstream, bool capture_stacks = false)
    : capture_stacks_{capture_stacks}, allocated_bytes_{0}, upstream_{upstream}
  {
  }

  /**
   * @brief Construct a new tracking resource adaptor using `upstream` to satisfy
   * allocation requests.
   *
   * @throws rmm::logic_error if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory
   * @param capture_stacks If true, capture stacks for allocation calls
   */
  tracking_resource_adaptor(Upstream* upstream, bool capture_stacks = false)
    : capture_stacks_{capture_stacks},
      allocated_bytes_{0},
      upstream_{to_device_async_resource_ref_checked(upstream)}
  {
  }

  tracking_resource_adaptor()                                 = delete;
  ~tracking_resource_adaptor() override                       = default;
  tracking_resource_adaptor(tracking_resource_adaptor const&) = delete;
  tracking_resource_adaptor(tracking_resource_adaptor&&) noexcept =
    default;  ///< @default_move_constructor
  tracking_resource_adaptor& operator=(tracking_resource_adaptor const&) = delete;
  tracking_resource_adaptor& operator=(tracking_resource_adaptor&&) noexcept =
    default;  ///< @default_move_assignment{tracking_resource_adaptor}

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept
  {
    return upstream_;
  }

  /**
   * @brief Get the outstanding allocations map
   *
   * @return std::map<void*, allocation_info> const& of a map of allocations. The key
   * is the allocated memory pointer and the data is the allocation_info structure, which
   * contains size and, potentially, stack traces.
   */
  std::map<void*, allocation_info> const& get_outstanding_allocations() const noexcept
  {
    return allocations_;
  }

  /**
   * @brief Query the number of bytes that have been allocated. Note that
   * this can not be used to know how large of an allocation is possible due
   * to both possible fragmentation and also internal page sizes and alignment
   * that is not tracked by this allocator.
   *
   * @return std::size_t number of bytes that have been allocated through this
   * allocator.
   */
  std::size_t get_allocated_bytes() const noexcept { return allocated_bytes_; }

  /**
   * @brief Gets a string containing the outstanding allocation pointers, their
   * size, and optionally the stack trace for when each pointer was allocated.
   *
   * Stack traces are only included if this resource adaptor was created with
   * `capture_stack == true`. Otherwise, outstanding allocation pointers will be
   * shown with their size and empty stack traces.
   *
   * @return std::string Containing the outstanding allocation pointers.
   */
  std::string get_outstanding_allocations_str() const
  {
    read_lock_t lock(mtx_);

    std::ostringstream oss;

    if (!allocations_.empty()) {
      for (auto const& alloc : allocations_) {
        oss << alloc.first << ": " << alloc.second.allocation_size << " B";
        if (alloc.second.strace != nullptr) {
          oss << " : callstack:" << std::endl << *alloc.second.strace;
        }
        oss << std::endl;
      }
    }

    return oss.str();
  }

  /**
   * @brief Log any outstanding allocations via RMM_LOG_DEBUG
   *
   */
  void log_outstanding_allocations() const
  {
#if RMM_LOG_ACTIVE_LEVEL <= RMM_LOG_LEVEL_DEBUG
    RMM_LOG_DEBUG("Outstanding Allocations: %s", get_outstanding_allocations_str());
#endif  // RMM_LOG_ACTIVE_LEVEL <= RMM_LOG_LEVEL_DEBUG
  }

 private:
  /**
   * @brief Allocates memory of size at least `bytes` using the upstream
   * resource as long as it fits inside the allocation limit.
   *
   * The returned pointer has at least 256B alignment.
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
    void* ptr = get_upstream_resource().allocate_async(bytes, stream);
    // track it.
    {
      write_lock_t lock(mtx_);
      allocations_.emplace(ptr, allocation_info{bytes, capture_stacks_});
    }
    allocated_bytes_ += bytes;

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
    get_upstream_resource().deallocate_async(ptr, bytes, stream);
    {
      write_lock_t lock(mtx_);

      const auto found = allocations_.find(ptr);

      // Ensure the allocation is found and the number of bytes match
      if (found == allocations_.end()) {
        // Don't throw but log an error. Throwing in a destructor (or any noexcept) will call
        // std::terminate
        RMM_LOG_ERROR(
          "Deallocating a pointer that was not tracked. Ptr: %p [%zuB], Current Num. Allocations: "
          "%zu",
          ptr,
          bytes,
          this->allocations_.size());
      } else {
        auto const allocated_bytes = found->second.allocation_size;

        allocations_.erase(found);

        if (allocated_bytes != bytes) {
          // Don't throw but log an error. Throwing in a destructor (or any noexcept) will call
          // std::terminate
          RMM_LOG_ERROR(
            "Alloc bytes (%zu) and Dealloc bytes (%zu) do not match", allocated_bytes, bytes);

          bytes = allocated_bytes;
        }
      }
    }
    allocated_bytes_ -= bytes;
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
    auto cast = dynamic_cast<tracking_resource_adaptor<Upstream> const*>(&other);
    if (cast == nullptr) { return false; }
    return get_upstream_resource() == cast->get_upstream_resource();
  }

  bool capture_stacks_;                           // whether or not to capture call stacks
  std::map<void*, allocation_info> allocations_;  // map of active allocations
  std::atomic<std::size_t> allocated_bytes_;      // number of bytes currently allocated
  std::shared_mutex mutable mtx_;                 // mutex for thread safe access to allocations_
  device_async_resource_ref upstream_;            // the upstream resource used for satisfying
                                                  // allocation requests
};

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
