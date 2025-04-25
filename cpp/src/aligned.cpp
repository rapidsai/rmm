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

#include <rmm/aligned.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace rmm {

bool is_pow2(std::size_t value) noexcept { return (value != 0U) && ((value & (value - 1)) == 0U); }

bool is_supported_alignment(std::size_t alignment) noexcept { return is_pow2(alignment); }

std::size_t align_up(std::size_t value, std::size_t alignment) noexcept
{
  assert(is_supported_alignment(alignment));
  return (value + (alignment - 1)) & ~(alignment - 1);
}

std::size_t align_down(std::size_t value, std::size_t alignment) noexcept
{
  assert(is_supported_alignment(alignment));
  return value & ~(alignment - 1);
}

bool is_aligned(std::size_t value, std::size_t alignment) noexcept
{
  assert(is_supported_alignment(alignment));
  return value == align_down(value, alignment);
}

bool is_pointer_aligned(void* ptr, std::size_t alignment) noexcept
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return is_aligned(reinterpret_cast<std::uintptr_t>(ptr), alignment);
}

}  // namespace rmm
