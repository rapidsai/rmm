/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cstddef>
#include <mutex>
#include <shared_mutex>

namespace rmm::mr {
/**
 * @brief Resource that uses `Upstream` to allocate memory and tracks statistics
 * on memory allocations.
 *
 * An instance of this resource can be constructed with an existing, upstream
 * resource in order to satisfy allocation requests, but any existing
 * allocations will be untracked. Tracking statistics stores the current, peak
 * and total memory allocations for both the number of bytes and number of calls
 * to the memory resource. `statistics_resource_adaptor` is intended as a debug
 * adaptor and shouldn't be used in performance-sensitive code.
 *
 * @tparam Upstream Type of the upstream resource used for
 * allocation/deallocation.
 */
template <typename Upstream>
class statistics_resource_adaptor final : public device_memory_resource {
 public:
  // can be a std::shared_mutex once C++17 is adopted
  using read_lock_t  = std::shared_lock<std::shared_timed_mutex>;
  using write_lock_t = std::unique_lock<std::shared_timed_mutex>;

  /**
   * @brief Utility struct for counting the current, peak, and total value of a number
   */
  struct counter {
    int64_t value{0};  // Current value
    int64_t peak{0};   // Max value of `value`
    int64_t total{0};  // Sum of all added values

    counter& operator+=(int64_t val)
    {
      value += val;
      total += val;
      peak = std::max(value, peak);
      return *this;
    }

    counter& operator-=(int64_t val)
    {
      value -= val;
      return *this;
    }
  };

  /**
   * @brief Construct a new statistics resource adaptor using `upstream` to satisfy
   * allocation requests.
   *
   * @throws `rmm::logic_error` if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory
   */
  statistics_resource_adaptor(Upstream* upstream) : upstream_{upstream}
  {
    RMM_EXPECTS(nullptr != upstream, "Unexpected null upstream resource pointer.");
  }

  statistics_resource_adaptor()                                   = delete;
  ~statistics_resource_adaptor() override                         = default;
  statistics_resource_adaptor(statistics_resource_adaptor const&) = delete;
  statistics_resource_adaptor& operator=(statistics_resource_adaptor const&) = delete;
  statistics_resource_adaptor(statistics_resource_adaptor&&) noexcept        = default;
  statistics_resource_adaptor& operator=(statistics_resource_adaptor&&) noexcept = default;

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
   * @brief Returns a `counter` struct for this adaptor containing the current,
   * peak, and total number of allocated bytes for this
   * adaptor since it was created.
   *
   * @return counter struct containing bytes count
   */
  counter get_bytes_counter() const noexcept
  {
    read_lock_t lock(mtx_);

    return bytes_;
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

    return allocations_;
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
    void* ptr = upstream_->allocate(bytes, stream);

    // increment the stats
    {
      write_lock_t lock(mtx_);

      // Increment the allocation_count_ while we have the lock
      bytes_ += bytes;
      allocations_ += 1;
    }

    return ptr;
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
    upstream_->deallocate(ptr, bytes, stream);

    {
      write_lock_t lock(mtx_);

      // Decrement the current allocated counts.
      bytes_ -= bytes;
      allocations_ -= 1;
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
    if (this == &other) { return true; }
    auto cast = dynamic_cast<statistics_resource_adaptor<Upstream> const*>(&other);
    return cast != nullptr ? upstream_->is_equal(*cast->get_upstream())
                           : upstream_->is_equal(other);
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

  counter bytes_;                        // peak, current and total allocated bytes
  counter allocations_;                  // peak, current and total allocation count
  std::shared_timed_mutex mutable mtx_;  // mutex for thread safe access to allocations_
  Upstream* upstream_;  // the upstream resource used for satisfying allocation requests
};

/**
 * @brief Convenience factory to return a `statistics_resource_adaptor` around the
 * upstream resource `upstream`.
 *
 * @tparam Upstream Type of the upstream `device_memory_resource`.
 * @param upstream Pointer to the upstream resource
 */
template <typename Upstream>
statistics_resource_adaptor<Upstream> make_statistics_adaptor(Upstream* upstream)
{
  return statistics_resource_adaptor<Upstream>{upstream};
}

}  // namespace rmm::mr
