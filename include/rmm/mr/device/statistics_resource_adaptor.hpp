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

#include <rmm/mr/device/device_memory_resource.hpp>
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
 * @brief Resource that uses `Upstream` to allocate memory and tracks statistics
 * on memory allocations.
 *
 * An instance of this resource can be constructed with an existing, upstream
 * resource in order to satisfy allocation requests, but any existing
 * allocations will be untracked. Tracking statistics stores the current, peak
 * and total memory allocations for both the number of bytes and number of calls
 * to the memory resource.
 *
 * This resource supports nested statistics, which makes it possible to track statistics
 * of a code block. Use `.push_counters()` to start tracking statistics on a code block
 * and use `.pop_counters()` to stop the tracking. The nested statistics are cascading
 * such that the statistics tracked by a code block include the statistics tracked in
 * all its tracked sub code blocks.
 *
 * `statistics_resource_adaptor` is intended as a debug adaptor and shouldn't be
 * used in performance-sensitive code.
 *
 * @tparam Upstream Type of the upstream resource used for
 * allocation/deallocation.
 */
template <typename Upstream>
class statistics_resource_adaptor final : public device_memory_resource {
 public:
  using read_lock_t =
    std::shared_lock<std::shared_mutex>;  ///< Type of lock used to synchronize read access
  using write_lock_t =
    std::unique_lock<std::shared_mutex>;  ///< Type of lock used to synchronize write access
  /**
   * @brief Utility struct for counting the current, peak, and total value of a number
   */
  struct counter {
    int64_t value{0};  ///< Current value
    int64_t peak{0};   ///< Max value of `value`
    int64_t total{0};  ///< Sum of all added values

    /**
     * @brief Add `val` to the current value and update the peak value if necessary
     *
     * @param val Value to add
     * @return Reference to this object
     */
    counter& operator+=(int64_t val)
    {
      value += val;
      total += val;
      peak = std::max(value, peak);
      return *this;
    }

    /**
     * @brief Subtract `val` from the current value and update the peak value if necessary
     *
     * @param val Value to subtract
     * @return Reference to this object
     */
    counter& operator-=(int64_t val)
    {
      value -= val;
      return *this;
    }

    /**
     * @brief Add `val` to the current value and update the peak value if necessary
     *
     * When updating the peak value, we assume that `val` is tracking a code block inside the
     * code block tracked by `this`. Because nested statistics are cascading, we have to convert
     * `val.peak` to the peak it would have been if it was part of the statistics tracked by `this`.
     * We do this by adding the current value that was active when `val` started tracking such that
     * we get `std::max(value + val.peak, peak)`.
     *
     * @param val Value to add
     */
    void add_counters_from_tracked_sub_block(const counter& val)
    {
      peak = std::max(value + val.peak, peak);
      value += val.value;
      total += val.total;
    }
  };

  /**
   * @brief Construct a new statistics resource adaptor using `upstream` to satisfy
   * allocation requests.
   *
   * @throws rmm::logic_error if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory
   */
  statistics_resource_adaptor(Upstream* upstream) : upstream_{upstream}
  {
    RMM_EXPECTS(nullptr != upstream, "Unexpected null upstream resource pointer.");
  }

  statistics_resource_adaptor()                                              = delete;
  ~statistics_resource_adaptor() override                                    = default;
  statistics_resource_adaptor(statistics_resource_adaptor const&)            = delete;
  statistics_resource_adaptor& operator=(statistics_resource_adaptor const&) = delete;
  statistics_resource_adaptor(statistics_resource_adaptor&&) noexcept =
    default;  ///< @default_move_constructor
  statistics_resource_adaptor& operator=(statistics_resource_adaptor&&) noexcept =
    default;  ///< @default_move_assignment{statistics_resource_adaptor}

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

  /**
   * @brief Returns a `counter` struct for this adaptor containing the current,
   * peak, and total number of allocated bytes for this
   * adaptor since it was created.
   *
   * @return counter struct containing bytes count
   */
  counter get_bytes_counter() const noexcept
  {
    read_lock_t lock(mtx_);

    return counter_stack_.top().first;
  }

  /**
   * @brief Returns a `counter` struct for this adaptor containing the current,
   * peak, and total number of allocation counts for this adaptor since it was
   * created.
   *
   * @return counter struct containing allocations count
   */
  counter get_allocations_counter() const noexcept
  {
    read_lock_t lock(mtx_);

    return counter_stack_.top().second;
  }

  /**
   * @brief Push a pair of zero counters on the stack, which becomes the new
   * counters returned by `get_bytes_counter()` and `get_allocations_counter()`
   *
   * @return top pair of counters <bytes, allocations> from the stack _before_
   * the push
   */
  std::pair<counter, counter> push_counters()
  {
    write_lock_t lock(mtx_);
    auto ret = counter_stack_.top();
    counter_stack_.push({counter{}, counter{}});
    return ret;
  }

  /**
   * @brief Pop a pair of counters from the stack
   *
   * @return top pair of counters <bytes, allocations> from the stack _before_
   * the pop
   * @throws std::out_of_range if the counter stack has fewer than two entries.
   */
  std::pair<counter, counter> pop_counters()
  {
    write_lock_t lock(mtx_);
    if (counter_stack_.size() < 2) { throw std::out_of_range("cannot pop the last counter pair"); }
    auto ret = counter_stack_.top();
    counter_stack_.pop();
    // Update the new top pair of counters
    counter_stack_.top().first.add_counters_from_tracked_sub_block(ret.first);
    counter_stack_.top().second.add_counters_from_tracked_sub_block(ret.second);
    return ret;
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
    void* ptr = upstream_->allocate(bytes, stream);

    // increment the stats
    {
      write_lock_t lock(mtx_);

      // Increment the allocation_count_ while we have the lock
      counter_stack_.top().first += bytes;
      counter_stack_.top().second += 1;
    }

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

    {
      write_lock_t lock(mtx_);

      // Decrement the current allocated counts.
      counter_stack_.top().first -= bytes;
      counter_stack_.top().second -= 1;
    }
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
    auto cast = dynamic_cast<statistics_resource_adaptor<Upstream> const*>(&other);
    if (cast == nullptr) { return upstream_->is_equal(other); }
    return get_upstream_resource() == cast->get_upstream_resource();
  }

  // Stack of counter pairs <bytes, allocations>
  // Invariant: the stack always contains at least one entry
  std::stack<std::pair<counter, counter>> counter_stack_{{std::make_pair(counter{}, counter{})}};
  std::shared_mutex mutable mtx_;  // mutex for thread safe access to allocations_
  Upstream* upstream_;             // the upstream resource used for satisfying allocation requests
};

/**
 * @brief Convenience factory to return a `statistics_resource_adaptor` around the
 * upstream resource `upstream`.
 *
 * @tparam Upstream Type of the upstream `device_memory_resource`.
 * @param upstream Pointer to the upstream resource
 * @return The new statistics resource adaptor
 */
template <typename Upstream>
statistics_resource_adaptor<Upstream> make_statistics_adaptor(Upstream* upstream)
{
  return statistics_resource_adaptor<Upstream>{upstream};
}

/** @} */  // end of group
}  // namespace rmm::mr
