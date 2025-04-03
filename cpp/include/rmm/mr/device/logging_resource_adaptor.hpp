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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/detail/format.hpp>
#include <rmm/logger.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <string>

namespace RMM_NAMESPACE {
namespace mr {

/**
 * @addtogroup device_resource_adaptors
 * @{
 * @file
 */
/**
 * @brief Resource that uses `Upstream` to allocate memory and logs information
 * about the requested allocation/deallocations.
 *
 * An instance of this resource can be constructed with an existing, upstream
 * resource in order to satisfy allocation requests and log
 * allocation/deallocation activity.
 *
 * @tparam Upstream Type of the upstream resource used for
 * allocation/deallocation.
 */
template <typename Upstream>
class logging_resource_adaptor final : public device_memory_resource {
 public:
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
   * @throws rmm::logic_error if `upstream == nullptr`
   * @throws spdlog::spdlog_ex if opening `filename` failed
   *
   * @param upstream The resource used for allocating/deallocating device memory
   * @param filename Name of file to write log info. If not specified, retrieves
   * the file name from the environment variable "RMM_LOG_FILE".
   * @param auto_flush If true, flushes the log for every (de)allocation. Warning, this will degrade
   * performance.
   */
  logging_resource_adaptor(Upstream* upstream,
                           std::string const& filename = get_default_filename(),
                           bool auto_flush             = false)
    : logging_resource_adaptor(to_device_async_resource_ref_checked(upstream), filename, auto_flush)
  {
  }

  /**
   * @brief Construct a new logging resource adaptor using `upstream` to satisfy
   * allocation requests and logging information about each allocation/free to
   * the ostream specified by `stream`.
   *
   * The logfile will be written using CSV formatting.
   *
   * @throws rmm::logic_error if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory
   * @param stream The ostream to write log info.
   * @param auto_flush If true, flushes the log for every (de)allocation. Warning, this will degrade
   * performance.
   */
  logging_resource_adaptor(Upstream* upstream, std::ostream& stream, bool auto_flush = false)
    : logging_resource_adaptor(to_device_async_resource_ref_checked(upstream), stream, auto_flush)
  {
  }

  /**
   * @brief Construct a new logging resource adaptor using `upstream` to satisfy
   * allocation requests and logging information about each allocation/free to
   * the ostream specified by `stream`.
   *
   * The logfile will be written using CSV formatting.
   *
   * @throws rmm::logic_error if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory
   * @param sinks A list of logging sinks to which log output will be written.
   * @param auto_flush If true, flushes the log for every (de)allocation. Warning, this will degrade
   * performance.
   */
  logging_resource_adaptor(Upstream* upstream,
                           std::initializer_list<rapids_logger::sink_ptr> sinks,
                           bool auto_flush = false)
    : logging_resource_adaptor{to_device_async_resource_ref_checked(upstream), sinks, auto_flush}
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
   * @param upstream The resource_ref used for allocating/deallocating device memory.
   * @param filename Name of file to write log info. If not specified, retrieves
   * the file name from the environment variable "RMM_LOG_FILE".
   * @param auto_flush If true, flushes the log for every (de)allocation. Warning, this will degrade
   * performance.
   */
  logging_resource_adaptor(device_async_resource_ref upstream,
                           std::string const& filename = get_default_filename(),
                           bool auto_flush             = false)
    : logging_resource_adaptor{make_logger(filename), upstream, auto_flush}
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
  logging_resource_adaptor(device_async_resource_ref upstream,
                           std::ostream& stream,
                           bool auto_flush = false)
    : logging_resource_adaptor{make_logger(stream), upstream, auto_flush}
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
   * @param sinks A list of logging sinks to which log output will be written.
   * @param auto_flush If true, flushes the log for every (de)allocation. Warning, this will degrade
   * performance.
   */
  logging_resource_adaptor(device_async_resource_ref upstream,
                           std::initializer_list<rapids_logger::sink_ptr> sinks,
                           bool auto_flush = false)
    : logging_resource_adaptor{make_logger(sinks), upstream, auto_flush}
  {
  }

  logging_resource_adaptor()                                           = delete;
  ~logging_resource_adaptor() override                                 = default;
  logging_resource_adaptor(logging_resource_adaptor const&)            = delete;
  logging_resource_adaptor& operator=(logging_resource_adaptor const&) = delete;
  logging_resource_adaptor(logging_resource_adaptor&&) noexcept =
    default;  ///< @default_move_constructor
  logging_resource_adaptor& operator=(logging_resource_adaptor&&) noexcept =
    default;  ///< @default_move_assignment{logging_resource_adaptor}

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept
  {
    return upstream_;
  }

  /**
   * @brief Flush logger contents.
   */
  void flush() { logger_->flush(); }

  /**
   * @brief Return the CSV header string
   *
   * @return CSV formatted header string of column names
   */
  [[nodiscard]] std::string header() const
  {
    return std::string{"Thread,Time,Action,Pointer,Size,Stream"};
  }

  /**
   * @brief Return the value of the environment variable RMM_LOG_FILE.
   *
   * @throws rmm::logic_error if `RMM_LOG_FILE` is not set.
   *
   * @return The value of RMM_LOG_FILE as `std::string`.
   */
  static std::string get_default_filename()
  {
    auto* filename = std::getenv("RMM_LOG_FILE");
    RMM_EXPECTS(filename != nullptr,
                "RMM logging requested without an explicit file name, but RMM_LOG_FILE is unset");
    return std::string{filename};
  }

 private:
  static auto make_logger(std::ostream& stream)
  {
    return std::make_shared<rapids_logger::logger>("RMM", stream);
  }

  static auto make_logger(std::string const& filename)
  {
    return std::make_shared<rapids_logger::logger>("RMM", filename);
  }

  static auto make_logger(std::initializer_list<rapids_logger::sink_ptr> sinks)
  {
    return std::make_shared<rapids_logger::logger>("RMM", sinks);
  }

  logging_resource_adaptor(std::shared_ptr<rapids_logger::logger> logger,
                           device_async_resource_ref upstream,
                           bool auto_flush)
    : logger_{logger}, upstream_{upstream}
  {
    if (auto_flush) { logger_->flush_on(rapids_logger::level_enum::info); }
    logger_->set_pattern("%v");
    logger_->info(header());
    logger_->set_pattern("%t,%H:%M:%S.%f,%v");
  }

  /**
   * @brief Allocates memory of size at least `bytes` using the upstream
   * resource and logs the allocation.
   *
   * If the upstream allocation is successful, logs the following CSV formatted
   * line to the file specified at construction:
   * ```
   * thread_id,*TIMESTAMP*,"allocate",*pointer*,*bytes*,*stream*
   * ```
   *
   * If the upstream allocation failed, logs the following CSV formatted line
   * to the file specified at construction:
   * ```
   * thread_id,*TIMESTAMP*,"allocate failure",0x0,*bytes*,*stream*
   * ```
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
    try {
      auto const ptr = get_upstream_resource().allocate_async(bytes, stream);
      logger_->info("allocate,%p,%zu,%s", ptr, bytes, rmm::detail::format_stream(stream));
      return ptr;
    } catch (...) {
      logger_->info(
        "allocate failure,%p,%zu,%s", nullptr, bytes, rmm::detail::format_stream(stream));
      throw;
    }
  }

  /**
   * @brief Free allocation of size `bytes` pointed to by `ptr` and log the
   * deallocation.
   *
   * Every invocation of `logging_resource_adaptor::do_deallocate` will write
   * the following CSV formatted line to the file specified at construction:
   * ```
   * thread_id,*TIMESTAMP*,"free",*bytes*,*stream*
   * ```
   *
   * @param ptr Pointer to be deallocated
   * @param bytes Size of the allocation
   * @param stream Stream on which to perform the deallocation
   */
  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) override
  {
    logger_->info("free,%p,%zu,%s", ptr, bytes, rmm::detail::format_stream(stream));
    get_upstream_resource().deallocate_async(ptr, bytes, stream);
  }

  /**
   * @brief Compare the upstream resource to another.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    if (this == &other) { return true; }
    auto const* cast = dynamic_cast<logging_resource_adaptor<Upstream> const*>(&other);
    if (cast == nullptr) { return false; }
    return get_upstream_resource() == cast->get_upstream_resource();
  }

  std::shared_ptr<rapids_logger::logger> logger_{};

  device_async_resource_ref upstream_;  ///< The upstream resource used for satisfying
                                        ///< allocation requests
};

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
