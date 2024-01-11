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

#include <cassert>
#include <cstddef>
#include <cstdint>

/**
 * @addtogroup utilities
 * @{
 */

namespace rmm {

/**
 * @brief Default alignment used for host memory allocated by RMM.
 *
 */
static constexpr std::size_t RMM_DEFAULT_HOST_ALIGNMENT{alignof(std::max_align_t)};

/**
 * @brief Default alignment used for CUDA memory allocation.
 *
 */
static constexpr std::size_t CUDA_ALLOCATION_ALIGNMENT{256};

/**
 * @brief Returns whether or not `n` is a power of 2.
 *
 */
constexpr bool is_pow2(std::size_t value) { return (value != 0U) && ((value & (value - 1)) == 0U); }

/**
 * @brief Returns whether or not `alignment` is a valid memory alignment.
 *
 */
constexpr bool is_supported_alignment(std::size_t alignment) { return is_pow2(alignment); }

/**
 * @brief Align up to nearest multiple of specified power of 2
 *
 * @param[in] value value to align
 * @param[in] alignment amount, in bytes, must be a power of 2
 *
 * @return Return the aligned value, as one would expect
 */
constexpr std::size_t align_up(std::size_t value, std::size_t alignment) noexcept
{
  assert(is_supported_alignment(alignment));
  return (value + (alignment - 1)) & ~(alignment - 1);
}

/**
 * @brief Align down to the nearest multiple of specified power of 2
 *
 * @param[in] value value to align
 * @param[in] alignment amount, in bytes, must be a power of 2
 *
 * @return Return the aligned value, as one would expect
 */
constexpr std::size_t align_down(std::size_t value, std::size_t alignment) noexcept
{
  assert(is_supported_alignment(alignment));
  return value & ~(alignment - 1);
}

/**
 * @brief Checks whether a value is aligned to a multiple of a specified power of 2
 *
 * @param[in] value value to check for alignment
 * @param[in] alignment amount, in bytes, must be a power of 2
 *
 * @return true if aligned
 */
constexpr bool is_aligned(std::size_t value, std::size_t alignment) noexcept
{
  assert(is_supported_alignment(alignment));
  return value == align_down(value, alignment);
}

inline bool is_pointer_aligned(void* ptr, std::size_t alignment = CUDA_ALLOCATION_ALIGNMENT)
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return is_aligned(reinterpret_cast<std::uintptr_t>(ptr), alignment);
}

}  // namespace rmm

/** @} */  // end of group
