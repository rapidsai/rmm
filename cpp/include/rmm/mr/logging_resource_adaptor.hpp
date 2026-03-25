/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/logging_resource_adaptor_impl.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <initializer_list>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

namespace RMM_NAMESPACE {
namespace mr {

/**
 * @addtogroup memory_resource_adaptors
 * @{
 * @file
 */

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
  : public cuda::mr::shared_resource<detail::logging_resource_adaptor_impl> {
  using shared_base = cuda::mr::shared_resource<detail::logging_resource_adaptor_impl>;

 public:
  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `logging_resource_adaptor` provides device accessible memory
   */
  RMM_CONSTEXPR_FRIEND void get_property(logging_resource_adaptor const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

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
   * @tparam Upstream Type of the upstream resource (must be convertible to
   * `cuda::mr::any_resource<cuda::mr::device_accessible>`).
   * @param upstream The resource_ref used for allocating/deallocating device memory.
   * @param filename Name of file to write log info. If not specified, retrieves
   * the file name from the environment variable "RMM_LOG_FILE".
   * @param auto_flush If true, flushes the log for every (de)allocation. Warning, this will degrade
   * performance.
   */
  template <
    class Upstream,
    std::enable_if_t<!std::is_same_v<std::decay_t<Upstream>, logging_resource_adaptor>, int> = 0>
  logging_resource_adaptor(Upstream&& upstream,
                           std::string const& filename = get_default_filename(),
                           bool auto_flush             = false)
    : shared_base(cuda::mr::make_shared_resource<detail::logging_resource_adaptor_impl>(
        std::make_shared<rapids_logger::logger>("RMM", filename),
        cuda::mr::any_resource<cuda::mr::device_accessible>{std::forward<Upstream>(upstream)},
        auto_flush))
  {
  }

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
  logging_resource_adaptor(cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
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
  logging_resource_adaptor(cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
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
};

static_assert(cuda::mr::resource_with<logging_resource_adaptor, cuda::mr::device_accessible>,
              "logging_resource_adaptor does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
