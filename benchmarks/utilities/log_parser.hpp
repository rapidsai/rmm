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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either ex  ess or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <rmm/detail/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include "rapidcsv.h"

#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace rmm {
namespace detail {

enum class action : bool { ALLOCATE, FREE };

/**
 * @brief Represents an allocation event
 *
 */
struct event {
  event()             = default;
  event(event const&) = default;
  event(action a, std::size_t s, void const* p)
    : act{a}, size{s}, pointer{reinterpret_cast<uintptr_t>(p)}
  {
  }

  event(action a, std::size_t s, uintptr_t p) : act{a}, size{s}, pointer{p} {}

  event(std::size_t tid, action a, std::size_t sz, uintptr_t p, uintptr_t s)
    : thread_id{tid}, act{a}, size{sz}, pointer{p}, stream{s}
  {
  }

  event(std::size_t tid, action a, std::size_t sz, void* p, uintptr_t s)
    : event{tid, a, sz, reinterpret_cast<uintptr_t>(p), s}
  {
  }

  friend std::ostream& operator<<(std::ostream& os, event const& e);

  action act{};           ///< Indicates if the event is an allocation or a free
  std::size_t size{};     ///< The size of the memory allocated or freed
  uintptr_t pointer{};    ///< The pointer returned from an allocation, or the
                          ///< pointer freed
  std::size_t thread_id;  ///< ID of the thread that initiated the event
  uintptr_t stream;       ///< Numeric representation of the CUDA stream on which the event occurred
};

inline std::ostream& operator<<(std::ostream& os, event const& e)
{
  auto act_string = (e.act == action::ALLOCATE) ? "allocate" : "free";

  os << "Thread: " << e.thread_id << std::setw(9) << act_string
     << " Size: " << std::setw(std::numeric_limits<std::size_t>::digits10) << e.size << " Pointer: "
     << "0x" << std::hex << e.pointer << std::dec << " Stream: " << e.stream;
  return os;
}

inline uintptr_t hex_string_to_int(std::string const& s) { return std::stoll(s, nullptr, 16); }

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

  std::vector<std::size_t> tids     = csv.GetColumn<std::size_t>("Thread");
  std::vector<std::string> actions  = csv.GetColumn<std::string>("Action");
  std::vector<std::size_t> sizes    = csv.GetColumn<std::size_t>("Size");
  std::vector<std::string> pointers = csv.GetColumn<std::string>("Pointer");
  std::vector<uintptr_t> streams    = csv.GetColumn<uintptr_t>("Stream");

  auto const size_list = {tids.size(), actions.size(), pointers.size(), streams.size()};

  RMM_EXPECTS(std::all_of(std::begin(size_list),
                          std::end(size_list),
                          [size = sizes.size()](auto i) { return i == size; }),
              "Size mismatch in columns of parsed log.");

  std::vector<event> events(sizes.size());

  for (std::size_t i = 0; i < actions.size(); ++i) {
    auto const& a = actions[i];
    RMM_EXPECTS((a == "allocate") or (a == "free"), "Invalid action string.");
    auto act  = (a == "allocate") ? action::ALLOCATE : action::FREE;
    events[i] = event{tids[i], act, sizes[i], hex_string_to_int(pointers[i]), streams[i]};
  }
  return events;
}

}  // namespace detail
}  // namespace rmm
