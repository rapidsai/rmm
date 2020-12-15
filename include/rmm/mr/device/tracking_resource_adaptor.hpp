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

#include <map>
#include <mutex>
#include <rmm/detail/error.hpp>
#include <rmm/detail/stack_trace.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <shared_mutex>
#include <sstream>

namespace rmm {
namespace mr {
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
  // can be a std::shared_mutex once C++17 is adopted
  using read_lock_t  = std::shared_lock<std::shared_timed_mutex>;
  using write_lock_t = std::unique_lock<std::shared_timed_mutex>;

  /**
   * @brief Information stored about an allocation. Includes the size
   * and a stack trace if the `tracking_resource_adaptor` was initialized
   * to capture stacks.
   *
   */
  struct allocation_info {
    std::unique_ptr<rmm::detail::stack_trace> strace;
    std::size_t allocation_size;

    allocation_info() = delete;
    allocation_info(std::size_t size, bool capture_stack)
      : strace{[&]() {
          return capture_stack ? std::make_unique<rmm::detail::stack_trace>() : nullptr;
        }()},
        allocation_size{size} {};
  };

  /**
   * @brief Stores the current, peak, and total number of allocation and
   * allocated bytes
   */
  struct allocation_counts {
    ssize_t current_bytes{0};    // Current outstanding bytes
    ssize_t current_count{0};    // Current outstanding count
    ssize_t peak_bytes{0};       // Max value of current_bytes
    ssize_t peak_count{0};       // Max value of current_count
    std::size_t total_bytes{0};  // Total allocated bytes
    std::size_t total_count{0};  // Total allocated count

    /**
     * @brief Increments the current, peak and total by bytes
     *
     * @param bytes Number of bytes allocated
     */
    void increment_count(size_t bytes)
    {
      current_bytes += bytes;
      current_count += 1;

      peak_bytes = std::max(current_bytes, peak_bytes);
      peak_count = std::max(current_count, peak_count);

      total_bytes += bytes;
      total_count += 1;
    }

    /**
     * @brief Decrement the current bytes and count. Peak and total remain unchanged
     *
     * @param bytes Number of bytes deallocated
     */
    void decrement_count(size_t bytes)
    {
      current_bytes -= bytes;
      current_count -= 1;
    }
  };

  /**
   * @brief Construct a new tracking resource adaptor using `upstream` to satisfy
   * allocation requests.
   *
   * @throws `rmm::logic_error` if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory
   * @param capture_stacks If true, capture stacks for allocation calls
   */
  tracking_resource_adaptor(Upstream* upstream, bool capture_stacks = false)
    : capture_stacks_{capture_stacks}, upstream_{upstream}
  {
    RMM_EXPECTS(nullptr != upstream, "Unexpected null upstream resource pointer.");

    // Need to maintain at least one on the stack at all times
    allocation_count_stack_.push_back(allocation_counts());
  }

  tracking_resource_adaptor()                                 = delete;
  virtual ~tracking_resource_adaptor()                        = default;
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
  std::size_t get_allocated_bytes() const noexcept
  {
    return get_total_allocation_counts().current_bytes;
  }

  /**
   * @brief Returns an allocation_counts struct for this adaptor containing the
   * total current, peak, and total number of bytes and allocation count for
   * this adaptor regardless of any push/popped allocation_counts. Note: Because
   * its possible to change memory resources at any time while maintaining the
   * same upstream memory resource, its possible to have a negative allocation
   * bytes or count if the number of deallocate() calls is greater than the
   * number of allocate().
   *
   * @return allocation_counts struct containing the allocation number of bytes
   * and count info
   */
  allocation_counts get_total_allocation_counts() const noexcept
  {
    read_lock_t lock(mtx_);

    // Start by copying the head
    allocation_counts sum_counts = allocation_count_stack_.front();

    auto prev_curr_bytes = sum_counts.current_bytes;
    auto prev_curr_count = sum_counts.current_count;

    for (size_t i = 1; i < allocation_count_stack_.size(); i++) {
      // Sum the total and current
      sum_counts.current_bytes += allocation_count_stack_[i].current_bytes;
      sum_counts.current_count += allocation_count_stack_[i].current_count;

      sum_counts.total_bytes += allocation_count_stack_[i].total_bytes;
      sum_counts.total_count += allocation_count_stack_[i].total_count;

      // Peak works differently. For each, the max value is the previous stack's
      // `current_bytes` + the current stack's `peak_bytes`
      sum_counts.peak_bytes =
        std::max(sum_counts.peak_bytes, prev_curr_bytes + allocation_count_stack_[i].peak_bytes);
      sum_counts.peak_count =
        std::max(sum_counts.peak_count, prev_curr_count + allocation_count_stack_[i].peak_count);

      prev_curr_bytes = allocation_count_stack_[i].current_bytes;
      prev_curr_count = allocation_count_stack_[i].current_count;
    }

    return sum_counts;
  }

  /**
   * @brief Pushes a new allocation_counts struct onto the stack to allow
   * tracking memory allocations for a particular section without creating a new
   * memory resource. Call `pop_allocation_counts` to retrieve the
   * allocation_counts since this method was called.
   *
   * @return allocation_counts Returns the previous allocation_counts at the top
   * of the stack
   */
  allocation_counts push_allocation_counts()
  {
    write_lock_t lock(mtx_);

    // Copy the top of the stack
    allocation_counts counts = allocation_count_stack_.back();

    // Push a new one on the stack
    allocation_count_stack_.emplace_back(allocation_counts{});

    return counts;
  }

  /**
   * @brief Pops the top allocation_counts struct off the stack and returns the
   * current, peak and total allocations since `push_allocation_counts` was
   * called.
   *
   * @throws rmm::out_of_range exception if `allocation_count_stack_.size() <=
   * 1`
   *
   * @return allocation_counts Returns the allocation_counts struct since
   * `push_allocation_counts` was called
   */
  allocation_counts pop_allocation_counts()
  {
    read_lock_t lock(mtx);

    RMM_EXPECTS(allocation_count_stack_.size() > 1,
                rmm::out_of_range,
                "Attempted to pop only allocation_counts on stack.");

    // Copy the top of the stack
    allocation_counts counts = allocation_count_stack_.back();

    // Pop the stack
    if (not allocations.empty()) {

    // When popping the stack, the top needs to be rolled into the new top to keep the total values
    // correct
    auto& new_top = allocation_count_stack_.back();

    // Make sure to do peak first here
    new_top.peak_bytes = std::max(new_top.peak_bytes, new_top.current_bytes + counts.peak_bytes);
    new_top.peak_count = std::max(new_top.peak_count, new_top.current_count + counts.peak_count);

    // Sum the total and current
    new_top.current_bytes += counts.current_bytes;
    new_top.current_count += counts.current_count;

    new_top.total_bytes += counts.total_bytes;
    new_top.total_count += counts.total_count;

    return counts;
  }

  /**
   * @brief Gets a string containing the outstanding allocation pointers, their
   * size, and optionally the stack trace for when each pointer was allocated.
   *
   * @return std::string Containing the outstanding allocation pointers.
   */
  std::string get_outstanding_allocations_str() const
  {
    read_lock_t lock(mtx_);

    std::ostringstream oss;

    if (!allocations_.empty()) {
      for (auto const& al : allocations_) {
        oss << al.first << ": " << al.second.allocation_size << " B";
        if (al.second.strace != nullptr) {
          oss << " : callstack:" << std::endl << *al.second.strace;
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
#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
    RMM_LOG_DEBUG("Outstanding Allocations: {}", get_outstanding_allocations_str());
#endif  // SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
  }

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
    void* p = upstream_->allocate(bytes, stream);

    // track it.
    {
      write_lock_t lock(mtx_);
      allocations_.emplace(p, allocation_info{bytes, capture_stacks_});

      // Increment the allocation_count_ while we have the lock
      allocation_count_stack_.back().increment_count(bytes);
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
  void do_deallocate(void* p, std::size_t bytes, cuda_stream_view stream) override
  {
    upstream_->deallocate(p, bytes, stream);

    {
      write_lock_t lock(mtx_);

      const auto found = allocations_.find(p);

      // Ensure the allocation is found and the number of bytes match
      if (found == allocations_.end()) {
        // Don't throw but log an error. Throwing in a descructor (or any noexcept) will call
        // std::terminate
        RMM_LOG_ERROR(
          "Deallocating a pointer that was not tracked. Ptr: {:p} [{}B], Current Num. Allocations: "
          "{}",
          fmt::ptr(p),
          bytes,
          this->allocations_.size());
      } else {
        allocations_.erase(found);

        auto allocated_bytes = found->second.allocation_size;

        if (allocated_bytes != bytes) {
          // Don't throw but log an error. Throwing in a descructor (or any noexcept) will call
          // std::terminate
          RMM_LOG_ERROR(
            "Alloc bytes ({}) and Dealloc bytes ({}) do not match", allocated_bytes, bytes);

          bytes = allocated_bytes;
        }
      }

      // Decrement the current allocated counts.
      allocation_count_stack_.back().decrement_count(bytes);
    }
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
      auto cast = dynamic_cast<tracking_resource_adaptor<Upstream> const*>(&other);
      return cast != nullptr ? upstream_->is_equal(*cast->get_upstream())
                             : upstream_->is_equal(other);
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
  std::pair<std::size_t, std::size_t> do_get_mem_info(cuda_stream_view stream) const override
  {
    return upstream_->get_mem_info(stream);
  }

  bool capture_stacks_;                           // whether or not to capture call stacks
  std::map<void*, allocation_info> allocations_;  // map of active allocations
  std::vector<allocation_counts>
    allocation_count_stack_;  // Stack of allocation_counts structs to track memory allocation
                              // between push and pop
  std::shared_timed_mutex mutable mtx_;  // mutex for thread safe access to allocations_
  Upstream* upstream_;  // the upstream resource used for satisfying allocation requests
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
