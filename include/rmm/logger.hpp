/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

// TODO: Add RMM_EXPORT tags or equivalent
// TODO: Remove, just here so vim shows everything
#define SUPPORTS_LOGGING
#define LOGGER_NAMESPACE rmm

// TODO: Remove this, right now it's just here so that logger() can return a
// spdlog::logger for backwards-compat testing.
#include <spdlog/spdlog.h>

#include <memory>
#include <string>

namespace LOGGER_NAMESPACE {

namespace detail {

/**
 * @class fake_logger_impl
 * @brief The fake implementation of the logger that performs no-ops.
 * This is the default behavior if real logging is not enabled.
 */
class fake_impl {
 public:
  fake_impl() = default;

  template <typename... Args>
  void log(Args&&... args)
  {
  }
};

// Forward declaration of the real implementation
struct impl;

}  // namespace detail

// These defines must be kept in sync with spdlog or bad things will happen!
#define RMM_LEVEL_TRACE    0
#define RMM_LEVEL_DEBUG    1
#define RMM_LEVEL_INFO     2
#define RMM_LEVEL_WARN     3
#define RMM_LEVEL_ERROR    4
#define RMM_LEVEL_CRITICAL 5
#define RMM_LEVEL_OFF      6

enum class level_enum : int32_t {
  trace    = RMM_LEVEL_TRACE,
  debug    = RMM_LEVEL_DEBUG,
  info     = RMM_LEVEL_INFO,
  warn     = RMM_LEVEL_WARN,
  error    = RMM_LEVEL_ERROR,
  critical = RMM_LEVEL_CRITICAL,
  off      = RMM_LEVEL_OFF,
  n_levels
};

/**
 * @class logger
 * @brief A logger class that either uses the real implementation (via spdlog) or performs no-ops if
 * not supported.
 */
class logger {
 public:
  /**
   * @brief Constructor for logger.
   * Initializes the logger based on whether logging is supported.
   */
#ifdef SUPPORTS_LOGGING
  logger(std::string name, std::string filename);
#else
  logger(std::string name, std::string filename) {}
#endif

  // Not default constructible.
  inline logger() = delete;

  // TODO: Remove inline, see below
  /**
   * @brief Destructor for logger.
   */
#ifdef SUPPORTS_LOGGING
  inline ~logger();
#else
  inline ~logger() = default;
#endif

  /**
   * @brief Copy constructor for logger.
   */
  logger(logger const&) = delete;
  // delete copy assignment operator
  logger& operator=(logger const&) = delete;
  // TODO: These functions shouldn't be inline, but are for the moment until
  // we switch over to a compiled component for the impl.
  // default move constructor
  inline logger(logger&&) = default;
  // default move assignment operator
  inline logger& operator=(logger&&) = default;

  template <typename... Args>
  void log(level_enum lvl, std::string const& format, Args&&... args)
  {
    auto size = static_cast<size_t>(std::snprintf(nullptr, 0, format.c_str(), args...) + 1);
    if (size <= 0) { throw std::runtime_error("Error during formatting."); }
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    log(lvl, {buf.get(), buf.get() + size - 1});
  }

  // TODO: Remove inline, see above.
  inline void log(level_enum lvl, std::string const& message);

  template <typename... Args>
  void trace(std::string const& format, Args&&... args)
  {
    log(level_enum::trace, format, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void debug(std::string const& format, Args&&... args)
  {
    log(level_enum::debug, format, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void info(std::string const& format, Args&&... args)
  {
    log(level_enum::info, format, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void warn(std::string const& format, Args&&... args)
  {
    log(level_enum::warn, format, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void error(std::string const& format, Args&&... args)
  {
    log(level_enum::error, format, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void critical(std::string const& format, Args&&... args)
  {
    log(level_enum::critical, format, std::forward<Args>(args)...);
  }

  /**
   * @brief Check at compile-time whether logging is supported.
   * @return `true` if logging is supported, `false` otherwise.
   */
  static constexpr bool supports_logging()
  {
#ifdef SUPPORTS_LOGGING
    return true;
#else
    return false;
#endif
  }

// TODO: Make this private once we don't need to access the impl for
// backwards-compat with legacy rmm.
// private:
// TODO: Support args to the impl constructor
#ifdef SUPPORTS_LOGGING
  std::unique_ptr<detail::impl> pImpl{};
#else
  std::unique_ptr<detail::fake_impl> pImpl{};
#endif
};

namespace detail {

#ifdef SUPPORTS_LOGGING
inline logger& default_logger();
#else
inline logger& default_logger()
{
  // This is a no-op so pass empty args.
  static class logger logger {
    "", ""
  };
  return logger;
}
#endif

// TODO: This only exists for backwards compat and should eventually be removed.
#ifdef SUPPORTS_LOGGING
inline spdlog::logger& logger();
#else
inline spdlog::logger& logger()
{
  // This branch won't compile. It's not worth supporting since it's not a
  // real backwards-compat path.
}
#endif

}  // namespace detail

inline logger& default_logger() { return detail::default_logger(); }

[[deprecated(
  "Support for direct access to spdlog loggers in rmm is planned for "
  "removal")]] inline spdlog::logger&
logger()
{
  return detail::logger();
}

// Macros for easier logging, similar to spdlog.
// TODO: Assumes that we want to respect spdlog's own logging macro settings.
// TODO: We need a way to rename these from RMM to something else. I don't know
// if that can be done in the code, though, and we might have to do it in the
// build system via configure_file.
// TODO: Should we switch this to use _LOGGER_ instead of _LOG_ to match SPDLOG
// instead of rmm? If we do that will be a breaking change for rmm.
// TODO: Should we support other signatures for log?
#define RMM_LOGGER_CALL(logger, level, ...) (logger).log(level, __VA_ARGS__)

// TODO: Need to define our own levels to map to spdlogs.
#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_TRACE
#define RMM_LOG_TRACE(...) \
  RMM_LOGGER_CALL(rmm::default_logger(), rmm::level_enum::trace, __VA_ARGS__)
#else
#define RMM_LOG_TRACE(...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
#define RMM_LOG_DEBUG(...) \
  RMM_LOGGER_CALL(rmm::default_logger(), rmm::level_enum::debug, __VA_ARGS__)
#else
#define RMM_LOG_DEBUG(...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_INFO
#define RMM_LOG_INFO(...) RMM_LOGGER_CALL(rmm::default_logger(), rmm::level_enum::info, __VA_ARGS__)
#else
#define RMM_LOG_INFO(...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_WARN
#define RMM_LOG_WARN(...) RMM_LOGGER_CALL(rmm::default_logger(), rmm::level_enum::warn, __VA_ARGS__)
#else
#define RMM_LOG_WARN(...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_ERROR
#define RMM_LOG_ERROR(...) \
  RMM_LOGGER_CALL(rmm::default_logger(), rmm::level_enum::error, __VA_ARGS__)
#else
#define RMM_LOG_ERROR(...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_CRITICAL
#define RMM_LOG_CRITICAL(...) \
  RMM_LOGGER_CALL(rmm::default_logger(), rmm::level_enum::critical, __VA_ARGS__)
#else
#define RMM_LOG_CRITICAL(...) (void)0
#endif

}  // namespace LOGGER_NAMESPACE

#include <rmm/logger_impl.hpp>
