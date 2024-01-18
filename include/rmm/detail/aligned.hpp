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

#include <rmm/aligned.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

namespace rmm::detail {

/**
 * @brief Default alignment used for host memory allocated by RMM.
 *
 */
[[deprecated("Use rmm::RMM_DEFAULT_HOST_ALIGNMENT instead.")]] static constexpr std::size_t
  RMM_DEFAULT_HOST_ALIGNMENT{rmm::RMM_DEFAULT_HOST_ALIGNMENT};

/**
 * @brief Default alignment used for CUDA memory allocation.
 *
 */
[[deprecated("Use rmm::CUDA_ALLOCATION_ALIGNMENT instead.")]] static constexpr std::size_t
  CUDA_ALLOCATION_ALIGNMENT{rmm::CUDA_ALLOCATION_ALIGNMENT};

/**
 * @brief Returns whether or not `n` is a power of 2.
 *
 */
[[deprecated("Use rmm::is_pow2 instead.")]] constexpr bool is_pow2(std::size_t value) noexcept
{
  return rmm::is_pow2(value);
}

/**
 * @brief Returns whether or not `alignment` is a valid memory alignment.
 *
 */
[[deprecated("Use rmm::is_supported_alignment instead.")]] constexpr bool is_supported_alignment(
  std::size_t alignment) noexcept
{
  return rmm::is_pow2(alignment);
}

/**
 * @brief Align up to nearest multiple of specified power of 2
 *
 * @param[in] value value to align
 * @param[in] alignment amount, in bytes, must be a power of 2
 *
 * @return Return the aligned value, as one would expect
 */
[[deprecated("Use rmm::align_up instead.")]] constexpr std::size_t align_up(
  std::size_t value, std::size_t alignment) noexcept
{
  return rmm::align_up(value, alignment);
}

/**
 * @brief Align down to the nearest multiple of specified power of 2
 *
 * @param[in] value value to align
 * @param[in] alignment amount, in bytes, must be a power of 2
 *
 * @return Return the aligned value, as one would expect
 */
[[deprecated("Use rmm::align_down instead.")]] constexpr std::size_t align_down(
  std::size_t value, std::size_t alignment) noexcept
{
  return rmm::align_down(value, alignment);
}

/**
 * @brief Checks whether a value is aligned to a multiple of a specified power of 2
 *
 * @param[in] value value to check for alignment
 * @param[in] alignment amount, in bytes, must be a power of 2
 *
 * @return true if aligned
 */
[[deprecated("Use rmm::is_aligned instead.")]] constexpr bool is_aligned(
  std::size_t value, std::size_t alignment) noexcept
{
  return rmm::is_aligned(value, alignment);
}

[[deprecated("Use rmm::is_pointer_aligned instead.")]] inline bool is_pointer_aligned(
  void* ptr, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
{
  return rmm::is_pointer_aligned(ptr, alignment);
}

/**
 * @brief Allocates sufficient host-accessible memory to satisfy the requested size `bytes` with
 * alignment `alignment` using the unary callable `alloc` to allocate memory.
 *
 * Given a pointer `p` to an allocation of size `n` returned from the unary callable `alloc`, the
 * pointer `q` returned from `aligned_alloc` points to a location within the `n` bytes with
 * sufficient space for `bytes` that satisfies `alignment`.
 *
 * In order to retrieve the original allocation pointer `p`, the offset between `p` and `q` is
 * stored at `q - sizeof(std::ptrdiff_t)`.
 *
 * Allocations returned from `aligned_host_allocate` *MUST* be freed by calling
 * `aligned_host_deallocate` with the same arguments for `bytes` and `alignment` with a compatible
 * unary `dealloc` callable capable of freeing the memory returned from `alloc`.
 *
 * If `alignment` is not a power of 2, behavior is undefined.
 * If `Alloc` does not allocate host-accessible memory, behavior is undefined.
 *
 * @param bytes The desired size of the allocation
 * @param alignment Desired alignment of allocation
 * @param alloc Unary callable given a size `n` will allocate at least `n` bytes
 * of host-accessible memory.
 * @tparam Alloc a unary callable type that allocates host-accessible memory.
 * @return void* Pointer into allocation of at least `bytes` with desired
 * `alignment`.
 */
template <typename Alloc>
void* aligned_host_allocate(std::size_t bytes, std::size_t alignment, Alloc alloc)
{
  assert(rmm::is_supported_alignment(alignment));

  // allocate memory for bytes, plus potential alignment correction,
  // plus store of the correction offset
  std::size_t padded_allocation_size{bytes + alignment + sizeof(std::ptrdiff_t)};

  char* const original = static_cast<char*>(alloc(padded_allocation_size));

  // account for storage of offset immediately prior to the aligned pointer
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  void* aligned{original + sizeof(std::ptrdiff_t)};

  // std::align modifies `aligned` to point to the first aligned location
  std::align(alignment, bytes, aligned, padded_allocation_size);

  // Compute the offset between the original and aligned pointers
  std::ptrdiff_t offset = static_cast<char*>(aligned) - original;

  // Store the offset immediately before the aligned pointer
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  *(static_cast<std::ptrdiff_t*>(aligned) - 1) = offset;

  return aligned;
}

/**
 * @brief Frees an allocation of host-accessible returned from `aligned_host_allocate`.
 *
 * Allocations returned from `aligned_host_allocate` *MUST* be freed by calling
 * `aligned_host_deallocate` with the same arguments for `bytes` and `alignment` with a compatible
 * unary `dealloc` callable capable of freeing the memory returned from `alloc`.
 *
 * @param p The aligned pointer to deallocate
 * @param bytes The number of bytes requested from `aligned_host_allocate`
 * @param alignment The alignment required from `aligned_host_allocate`
 * @param dealloc A unary callable capable of freeing host-accessible memory returned from `alloc`
 * in `aligned_host_allocate`.
 * @tparam Dealloc A unary callable type that deallocates host-accessible memory.
 */
template <typename Dealloc>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void aligned_host_deallocate(void* ptr,
                             [[maybe_unused]] std::size_t bytes,
                             [[maybe_unused]] std::size_t alignment,
                             Dealloc dealloc) noexcept
{
  assert(rmm::is_supported_alignment(alignment));

  // Get offset from the location immediately prior to the aligned pointer
  // NOLINTNEXTLINE
  std::ptrdiff_t const offset = *(reinterpret_cast<std::ptrdiff_t*>(ptr) - 1);

  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  void* const original = static_cast<char*>(ptr) - offset;

  dealloc(original);
}
}  // namespace rmm::detail
