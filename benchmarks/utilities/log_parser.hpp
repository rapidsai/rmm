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

#include <thrust/iterator/zip_iterator.h>
#include <memory>
#include <stdexcept>
#include <string>
#include "rapidcsv.h"

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

  action act{};           ///< Indicates if the event is an allocation or a free
  std::size_t size{};     ///< The size of the memory allocated or freed
  uintptr_t pointer{};    ///< The pointer returned from an allocation, or the
                          ///< pointer freed
  std::string thread_id;  ///< ID of the thread that initiated the event
  uintptr_t stream;       ///< Numeric representation of the CUDA stream on which the event occurred
};

bool operator==(event const& lhs, event const& rhs)
{
  return std::tie(lhs.act, lhs.size, lhs.pointer) == std::tie(rhs.act, rhs.size, rhs.pointer);
}

uintptr_t hex_string_to_int(std::string const& s) { return std::stoll(s, nullptr, 16); }

/**
 * @brief Parses a RMM log file into a vector of events
 *
 * Parses a log file generated from `rmm::mr::logging_resource_adaptor` into a vector of `event`s.
 * An `event` describes an allocation/deallocation event that occurred via the logging adaptor.
 *
 * @param filename Name of the RMM log file
 * @return Vector of events from the contents of the log file
 */
std::vector<event> parse_csv(std::string const& filename)
{
  rapidcsv::Document csv(filename, rapidcsv::LabelParams(0, -1));

  std::vector<std::string> tids     = csv.GetColumn<std::string>("Thread");
  std::vector<std::string> actions  = csv.GetColumn<std::string>("Action");
  std::vector<std::size_t> sizes    = csv.GetColumn<std::size_t>("Size");
  std::vector<std::string> pointers = csv.GetColumn<std::string>("Pointer");

  if ((sizes.size() != actions.size()) or (sizes.size() != pointers.size())) {
    throw std::runtime_error{"Size mismatch in actions, sizes, or pointers."};
  }

  std::vector<event> events(sizes.size());

  for (std::size_t i = 0; i < actions.size(); ++i) {
    auto act  = (actions[i] == "allocate") ? action::ALLOCATE : action::FREE;
    events[i] = event{act, sizes[i], hex_string_to_int(pointers[i])};
  }
  return events;
}

}  // namespace detail
}  // namespace rmm