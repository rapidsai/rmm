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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either ex  ess or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "rapidcsv.h"

#include <rmm/detail/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

namespace rmm::detail {

enum class action { ALLOCATE, FREE, ALLOCATE_FAILURE };

/**
 * @brief Represents an allocation event
 *
 */
struct event {
  event()                            = default;
  event(event const&)                = default;
  event& operator=(event const&)     = default;
  event(event&&) noexcept            = default;
  event& operator=(event&&) noexcept = default;
  ~event()                           = default;
  event(action act, std::size_t size, void const* ptr)
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    : act{act}, size{size}, pointer{reinterpret_cast<uintptr_t>(ptr)}
  {
  }

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  event(action act, std::size_t size, uintptr_t ptr) : act{act}, size{size}, pointer{ptr} {}

  event(std::size_t tid,
        action act,
        std::size_t size,  // NOLINT(bugprone-easily-swappable-parameters)
        uintptr_t ptr,
        uintptr_t stream,
        std::size_t index)
    : act{act}, size{size}, pointer{ptr}, thread_id{tid}, stream{stream}, index{index}
  {
  }

  event(
    std::size_t tid, action act, std::size_t size, void* ptr, uintptr_t stream, std::size_t index)
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    : event{tid, act, size, reinterpret_cast<uintptr_t>(ptr), stream, index}
  {
  }

  friend std::ostream& operator<<(std::ostream& os, event const& evt);

  action act{};             ///< Indicates if the event is an allocation or a free
  std::size_t size{};       ///< The size of the memory allocated or freed
  uintptr_t pointer{};      ///< The pointer returned from an allocation, or the
                            ///< pointer freed
  std::size_t thread_id{};  ///< ID of the thread that initiated the event
  uintptr_t stream{};   ///< Numeric representation of the CUDA stream on which the event occurred
  std::size_t index{};  ///< Original ordering index of the event
};

inline std::ostream& operator<<(std::ostream& os, event const& evt)
{
  const auto* act_string{[&evt] {
    switch (evt.act) {
      case action::ALLOCATE: return "allocate";
      case action::FREE: return "free";
      default: return "allocate failure";
    }
  }()};
  const auto format_width{9};

  os << "Thread: " << evt.thread_id << std::setw(format_width) << act_string
     << " Size: " << std::setw(std::numeric_limits<std::size_t>::digits10) << evt.size
     << " Pointer: "
     << "0x" << std::hex << evt.pointer << std::dec << " Stream: " << evt.stream;
  return os;
}

/**
 * @brief Parse a log timestamp into a std::chrono::time_point
 *
 * @note currently unused. Seemed necessary for ordering but it appears the log currently
 * is in timestamp order even for multithreaded logs.
 * @note This function can be simplified with C++20 and later.
 *
 * @param str_time The input time in format "HH:MM:SS:us" where us is a 6 digits microseconds part
 * of the current second. (This is the format rmm::mr::logging_resource_adaptor outputs)
 * @return std::chrono::time_point<std::chrono::system_clock> Converted time point.
 */
inline std::chrono::time_point<std::chrono::system_clock> parse_time(std::string const& str_time)
{
  std::size_t current  = str_time.find(':');
  std::size_t previous = 0;
  int hours            = std::stoi(str_time.substr(previous, current - previous));
  previous             = current;
  current              = str_time.find(':');
  int minutes          = std::stoi(str_time.substr(previous, current - previous));
  previous             = current;
  current              = str_time.find(':');
  int seconds          = std::stoi(str_time.substr(previous, current - previous));
  int microseconds     = std::stoi(str_time.substr(current + 1, str_time.length()));

  auto const epoch_year{1970};
  std::tm time{seconds, minutes, hours, 1, 0, epoch_year, 0, 0, 0};

  auto timepoint = std::chrono::system_clock::from_time_t(std::mktime(&time));
  timepoint += std::chrono::microseconds{microseconds};
  return timepoint;
}

/**
 * @brief Parses a RMM log file into a vector of events
 *
 * Parses a log file generated from `rmm::mr::logging_resource_adaptor` into a vector of `event`s.
 * An `event` describes an allocation/deallocation event that occurred via the logging adaptor.
 *
 * @param filename Name of the RMM log file
 * @return Vector of events from the contents of the log file
 */
inline std::vector<event> parse_csv(std::string const& filename)
{
  rapidcsv::Document csv(filename, rapidcsv::LabelParams(0, -1));

  std::vector<std::size_t> tids    = csv.GetColumn<std::size_t>("Thread");
  std::vector<std::string> actions = csv.GetColumn<std::string>("Action");

  auto parse_pointer = [](std::string const& str, uintptr_t& ptr) {
    auto const base{16};
    ptr = (str == "(nil)") ? 0 : std::stoll(str, nullptr, base);
  };

  std::vector<uintptr_t> pointers = csv.GetColumn<uintptr_t>("Pointer", parse_pointer);
  std::vector<std::size_t> sizes  = csv.GetColumn<std::size_t>("Size");
  std::vector<uintptr_t> streams  = csv.GetColumn<uintptr_t>("Stream");

  auto const size_list = {tids.size(), actions.size(), pointers.size(), streams.size()};

  RMM_EXPECTS(std::all_of(std::begin(size_list),
                          std::end(size_list),
                          [size = sizes.size()](auto val) { return val == size; }),
              "Size mismatch in columns of parsed log.");

  std::vector<event> events(sizes.size());

  for (std::size_t i = 0; i < actions.size(); ++i) {
    auto const& action = actions[i];
    RMM_EXPECTS((action == "allocate") or (action == "allocate failure") or (action == "free"),
                "Invalid action string.");
    auto act{action::ALLOCATE_FAILURE};
    if (action == "allocate") {
      act = action::ALLOCATE;
    } else if (action == "free") {
      act = action::FREE;
    }
    events[i] = event{tids[i], act, sizes[i], pointers[i], streams[i], i};
  }
  return events;
}

}  // namespace rmm::detail
