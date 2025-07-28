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

#include <rmm/error.hpp>

#include <string>

namespace rmm {

bad_alloc::bad_alloc(const char* msg) : _what{std::string{std::bad_alloc::what()} + ": " + msg} {}

bad_alloc::bad_alloc(std::string const& msg) : bad_alloc{msg.c_str()} {}

const char* bad_alloc::what() const noexcept { return _what.c_str(); }

out_of_memory::out_of_memory(const char* msg) : bad_alloc{std::string{"out_of_memory: "} + msg} {}

out_of_memory::out_of_memory(std::string const& msg) : out_of_memory{msg.c_str()} {}

}  // namespace rmm
