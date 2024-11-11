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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/logger.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <fmt/core.h>
#ifdef RMM_BACKWARDS_COMPATIBILITY
#include <spdlog/common.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>
#endif

#include <cstddef>
#include <memory>
#include <sstream>
#include <string_view>

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
#ifdef RMM_BACKWARDS_COMPATIBILITY
  logging_resource_adaptor(Upstream* upstream,
                           spdlog::sinks_init_list sinks,
                           bool auto_flush = false)
    : logging_resource_adaptor{to_device_async_resource_ref_checked(upstream), sinks, auto_flush}
  {
  }
#else
  template <typename SinkPtr>
  logging_resource_adaptor(Upstream* upstream,
                           std::initializer_list<SinkPtr> sinks,
                           bool auto_flush = false)
    : logging_resource_adaptor{to_device_async_resource_ref_checked(upstream), sinks, auto_flush}
  {
  }
#endif

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
#ifdef RMM_BACKWARDS_COMPATIBILITY
  logging_resource_adaptor(device_async_resource_ref upstream,
                           spdlog::sinks_init_list sinks,
                           bool auto_flush = false)
    : logging_resource_adaptor{make_logger(sinks), upstream, auto_flush}
  {
  }
#else
  template <typename SinkPtr>
  logging_resource_adaptor(device_async_resource_ref upstream,
                           std::initializer_list<SinkPtr> sinks,
                           bool auto_flush = false)
    : logging_resource_adaptor{make_logger(sinks), upstream, auto_flush}
  {
  }
#endif

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
#ifdef RMM_BACKWARDS_COMPATIBILITY
    return std::make_shared<spdlog::logger>(
      "RMM", std::make_shared<spdlog::sinks::ostream_sink_mt>(stream));
#else
    return std::make_shared<logger>("RMM", stream);
#endif
  }

  static auto make_logger(std::string const& filename)
  {
#ifdef RMM_BACKWARDS_COMPATIBILITY
    return std::make_shared<spdlog::logger>(
      "RMM", std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, true /*truncate file*/));
#else
    return std::make_shared<logger>("RMM", filename);
#endif
  }

  // TODO: See if there is a way to make this function only valid for our sink
  // or spdlog's without leaking spdlog symbols. When logging isn't enabled our
  // sink type is constructible from anything, so that sort of analysis won't
  // help (and fixing that would require the same ideas as fixing this).
  template <typename SinkPtr>
  static auto make_logger(std::initializer_list<SinkPtr> sinks)
  {
#ifdef RMM_BACKWARDS_COMPATIBILITY
    return std::make_shared<spdlog::logger>("RMM", sinks);
#else
    // Support passing either
    if constexpr (std::is_same_v<SinkPtr, sink>) {
      return std::make_shared<logger>("RMM", sinks);
    } else {
      std::vector<std::shared_ptr<sink>> rmm_sinks;
      rmm_sinks.reserve(sinks.size());
      for (const auto& s : sinks) {
        rmm_sinks.push_back(std::make_shared<sink>(s));
      }
      return std::make_shared<logger>("RMM", std::move(rmm_sinks));
    }
#endif
  }

#ifdef RMM_BACKWARDS_COMPATIBILITY
  logging_resource_adaptor(std::shared_ptr<spdlog::logger> logger,
#else
  logging_resource_adaptor(std::shared_ptr<logger> logger,
#endif
                           device_async_resource_ref upstream,
                           bool auto_flush)
    : logger_{logger}, upstream_{upstream}
  {
#ifdef RMM_BACKWARDS_COMPATIBILITY
    if (auto_flush) { logger_->flush_on(spdlog::level::info); }
#else
    if (auto_flush) { logger_->flush_on(level_enum::info); }
#endif
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
#ifdef RMM_BACKWARDS_COMPATIBILITY
      logger_->info("allocate,{},{},{}", ptr, bytes, fmt::ptr(stream.value()));
#else
      logger_->info("allocate,%llx,%zu,%llx", ptr, bytes, stream.value());
#endif
      return ptr;
    } catch (...) {
#ifdef RMM_BACKWARDS_COMPATIBILITY
      logger_->info("allocate failure,{},{},{}", nullptr, bytes, fmt::ptr(stream.value()));
#else
      logger_->info("allocate failure,%llx,%zu,%llx", nullptr, bytes, stream.value());
#endif
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
#ifdef RMM_BACKWARDS_COMPATIBILITY
    logger_->info("free,{},{},{}", ptr, bytes, fmt::ptr(stream.value()));
#else
    logger_->info("free,%llx,%zu,%llx", ptr, bytes, stream.value());
#endif
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

#ifdef RMM_BACKWARDS_COMPATIBILITY
  std::shared_ptr<spdlog::logger> logger_;  ///< spdlog logger object
#else
  std::shared_ptr<logger> logger_{};
#endif

  device_async_resource_ref upstream_;  ///< The upstream resource used for satisfying
                                        ///< allocation requests
};

/**
 * @brief Convenience factory to return a `logging_resource_adaptor` around the
 * upstream resource `upstream`.
 *
 * @tparam Upstream Type of the upstream `device_memory_resource`.
 * @param upstream Pointer to the upstream resource
 * @param filename Name of the file to write log info. If not specified,
 * retrieves the log file name from the environment variable "RMM_LOG_FILE".
 * @param auto_flush If true, flushes the log for every (de)allocation. Warning, this will degrade
 * performance.
 * @return The new logging resource adaptor
 */
template <typename Upstream>
[[deprecated(
  "make_logging_adaptor is deprecated in RMM 24.10. Use the logging_resource_adaptor constructor "
  "instead.")]]
logging_resource_adaptor<Upstream> make_logging_adaptor(
  Upstream* upstream,
  std::string const& filename = logging_resource_adaptor<Upstream>::get_default_filename(),
  bool auto_flush             = false)
{
  return logging_resource_adaptor<Upstream>{upstream, filename, auto_flush};
}

/**
 * @brief Convenience factory to return a `logging_resource_adaptor` around the
 * upstream resource `upstream`.
 *
 * @tparam Upstream Type of the upstream `device_memory_resource`.
 * @param upstream Pointer to the upstream resource
 * @param stream The ostream to write log info.
 * @param auto_flush If true, flushes the log for every (de)allocation. Warning, this will degrade
 * performance.
 * @return The new logging resource adaptor
 */
template <typename Upstream>
[[deprecated(
  "make_logging_adaptor is deprecated in RMM 24.10. Use the logging_resource_adaptor constructor "
  "instead.")]]
logging_resource_adaptor<Upstream> make_logging_adaptor(Upstream* upstream,
                                                        std::ostream& stream,
                                                        bool auto_flush = false)
{
  return logging_resource_adaptor<Upstream>{upstream, stream, auto_flush};
}

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
