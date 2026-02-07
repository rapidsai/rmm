/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/logger.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <string>

namespace RMM_NAMESPACE {
namespace mr {

/**
 * @addtogroup memory_resource_adaptors
 * @{
 * @file
 */

namespace detail {

/**
 * @brief Implementation class for logging_resource_adaptor.
 *
 * This class satisfies the CCCL `cuda::mr::resource` concept and provides
 * the actual logging functionality. It is held by `logging_resource_adaptor`
 * via `cuda::mr::shared_resource` for reference-counted ownership.
 */
class logging_resource_adaptor_impl {
 public:
  /**
   * @brief Construct a logging resource adaptor impl.
   *
   * @param logger The logger to use for logging allocations/deallocations
   * @param upstream The upstream resource used for allocating/deallocating device memory
   * @param auto_flush If true, flushes the log for every (de)allocation
   */
  logging_resource_adaptor_impl(std::shared_ptr<rapids_logger::logger> logger,
                                device_async_resource_ref upstream,
                                bool auto_flush);

  /**
   * @brief Allocate memory synchronously.
   *
   * @param bytes The size in bytes of the allocation
   * @param alignment The alignment of the allocation
   * @return Pointer to the newly allocated memory
   */
  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t));

  /**
   * @brief Deallocate memory synchronously.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation
   * @param alignment The alignment of the allocation
   */
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept;

  /**
   * @brief Allocate memory asynchronously on a stream.
   *
   * @param stream The stream on which to perform the allocation
   * @param bytes The size in bytes of the allocation
   * @param alignment The alignment of the allocation
   * @return Pointer to the newly allocated memory
   */
  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t));

  /**
   * @brief Deallocate memory asynchronously on a stream.
   *
   * @param stream The stream on which to perform the deallocation
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation
   * @param alignment The alignment of the allocation
   */
  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept;

  /**
   * @brief Equality comparison.
   *
   * Two logging_resource_adaptor_impl instances are equal if they have the
   * same upstream resource AND the same logger.
   *
   * @param other The other impl to compare to
   * @return true If the two impls are equivalent
   * @return false If the two impls are not equivalent
   */
  bool operator==(logging_resource_adaptor_impl const& other) const noexcept
  {
    return upstream_ == other.upstream_ && logger_ == other.logger_;
  }

  /**
   * @brief Inequality comparison.
   *
   * @param other The other impl to compare to
   * @return true If the two impls are not equivalent
   * @return false If the two impls are equivalent
   */
  bool operator!=(logging_resource_adaptor_impl const& other) const noexcept
  {
    return !(*this == other);
  }

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept;

  /**
   * @brief Flush logger contents.
   */
  void flush();

  /**
   * @brief Return the CSV header string
   *
   * @return CSV formatted header string of column names
   */
  [[nodiscard]] std::string header() const;

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `logging_resource_adaptor_impl` provides device accessible memory
   */
  friend void get_property(logging_resource_adaptor_impl const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

 private:
  std::shared_ptr<rapids_logger::logger> logger_{};
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
};

}  // namespace detail

/**
 * @brief Resource that uses an upstream resource to allocate memory and logs information
 * about the requested allocation/deallocations.
 *
 * An instance of this resource can be constructed with an existing, upstream
 * resource in order to satisfy allocation requests and log
 * allocation/deallocation activity.
 *
 * This class is copyable and shares ownership of its internal state, allowing
 * multiple instances to safely reference the same underlying resource and logger.
 */
class RMM_EXPORT logging_resource_adaptor
  : public device_memory_resource,
    public cuda::mr::shared_resource<detail::logging_resource_adaptor_impl> {
  using shared_base = cuda::mr::shared_resource<detail::logging_resource_adaptor_impl>;

 public:
  // Begin legacy device_memory_resource compatibility layer
  using device_memory_resource::allocate;
  using device_memory_resource::allocate_sync;
  using device_memory_resource::deallocate;
  using device_memory_resource::deallocate_sync;

  /**
   * @brief Equality comparison operator.
   *
   * @param other The other logging_resource_adaptor to compare against.
   * @return true if both adaptors share the same underlying state.
   */
  [[nodiscard]] bool operator==(logging_resource_adaptor const& other) const noexcept
  {
    return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(other);
  }

  /**
   * @brief Inequality comparison operator.
   *
   * @param other The other logging_resource_adaptor to compare against.
   * @return true if the adaptors do not share the same underlying state.
   */
  [[nodiscard]] bool operator!=(logging_resource_adaptor const& other) const noexcept
  {
    return !(*this == other);
  }
  // End legacy device_memory_resource compatibility layer
  /**
   * @brief Construct a new logging resource adaptor using `upstream` to satisfy
   * allocation requests and logging information about each allocation/free to
   * the file specified by `filename`.
   *
   * The logfile will be written using CSV formatting.
   *
   * Clears the contents of `filename` if it already exists.
   *
   * Creating multiple `logging_resource_adaptor`s with the same `filename` will
   * result in undefined behavior.
   *
   * @throws spdlog::spdlog_ex if opening `filename` failed
   *
   * @param upstream The resource_ref used for allocating/deallocating device memory.
   * @param filename Name of file to write log info. If not specified, retrieves
   * the file name from the environment variable "RMM_LOG_FILE".
   * @param auto_flush If true, flushes the log for every (de)allocation. Warning, this will degrade
   * performance.
   */
  logging_resource_adaptor(device_async_resource_ref upstream,
                           std::string const& filename = get_default_filename(),
                           bool auto_flush             = false);

  /**
   * @brief Construct a new logging resource adaptor using `upstream` to satisfy
   * allocation requests and logging information about each allocation/free to
   * the ostream specified by `stream`.
   *
   * The logfile will be written using CSV formatting.
   *
   * @param upstream The resource_ref used for allocating/deallocating device memory.
   * @param stream The ostream to write log info.
   * @param auto_flush If true, flushes the log for every (de)allocation. Warning, this will degrade
   * performance.
   */
  logging_resource_adaptor(device_async_resource_ref upstream,
                           std::ostream& stream,
                           bool auto_flush = false);

  /**
   * @brief Construct a new logging resource adaptor using `upstream` to satisfy
   * allocation requests and logging information about each allocation/free to
   * the sinks specified.
   *
   * The logfile will be written using CSV formatting.
   *
   * @param upstream The resource_ref used for allocating/deallocating device memory.
   * @param sinks A list of logging sinks to which log output will be written.
   * @param auto_flush If true, flushes the log for every (de)allocation. Warning, this will degrade
   * performance.
   */
  logging_resource_adaptor(device_async_resource_ref upstream,
                           std::initializer_list<rapids_logger::sink_ptr> sinks,
                           bool auto_flush = false);

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept;

  /**
   * @brief Flush logger contents.
   */
  void flush();

  /**
   * @brief Return the CSV header string
   *
   * @return CSV formatted header string of column names
   */
  [[nodiscard]] std::string header() const;

  /**
   * @brief Return the value of the environment variable RMM_LOG_FILE.
   *
   * @throws rmm::logic_error if `RMM_LOG_FILE` is not set.
   *
   * @return The value of RMM_LOG_FILE as `std::string`.
   */
  static std::string get_default_filename();

  // Begin legacy device_memory_resource compatibility layer
 private:
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override;

  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) noexcept override;

  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override;
  // End legacy device_memory_resource compatibility layer
};

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
