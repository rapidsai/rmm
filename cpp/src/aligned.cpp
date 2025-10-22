/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
