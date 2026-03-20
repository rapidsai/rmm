/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/format.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstddef>

namespace rmm::mr {

/**
 * @brief A device memory resource that simulates a fix-sized GPU.
 *
 * Only allocation calls are simulated. New memory is allocated sequentially in monotonically
 * increasing address based on the requested size, until the predetermined size is exceeded.
 *
 * Deallocation calls are ignored.
 */
class simulated_memory_resource final {
 public:
  /**
   * @brief Construct a `simulated_memory_resource`.
   *
   * @param memory_size_bytes The size of the memory to simulate.
   */
  explicit simulated_memory_resource(std::size_t memory_size_bytes)
    : begin_{reinterpret_cast<char*>(0x100)},                    // NOLINT
      end_{reinterpret_cast<char*>(begin_ + memory_size_bytes)}  // NOLINT
  {
  }

  ~simulated_memory_resource() = default;

  simulated_memory_resource(simulated_memory_resource const&)            = default;
  simulated_memory_resource& operator=(simulated_memory_resource const&) = default;
  simulated_memory_resource(simulated_memory_resource&&)                 = default;
  simulated_memory_resource& operator=(simulated_memory_resource&&)      = default;

  /**
   * @brief Allocates memory of size at least `bytes`.
   *
   * @note Stream argument is ignored
   *
   * @throws rmm::bad_alloc if the requested allocation could not be fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate([[maybe_unused]] cuda::stream_ref stream,
                 std::size_t bytes,
                 [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    RMM_EXPECTS(begin_ + bytes <= end_,
                "Simulated memory size exceeded (failed to allocate " +
                  rmm::detail::format_bytes(bytes) + ")",
                rmm::bad_alloc);
    auto* ptr = static_cast<void*>(begin_);
    begin_ += bytes;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return ptr;
  }

  /**
   * @brief Deallocate memory pointed to by `ptr`.
   *
   * @note This call is ignored.
   */
  void deallocate([[maybe_unused]] cuda::stream_ref stream,
                  void* /*ptr*/,
                  std::size_t /*bytes*/,
                  [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate(cuda::stream_ref{reinterpret_cast<cudaStream_t>(0)}, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(cuda::stream_ref{reinterpret_cast<cudaStream_t>(0)}, ptr, bytes, alignment);
  }

  bool operator==(simulated_memory_resource const&) const noexcept { return true; }
  bool operator!=(simulated_memory_resource const&) const noexcept { return false; }

  friend void get_property(simulated_memory_resource const&, cuda::mr::device_accessible) noexcept
  {
  }

 private:
  char* begin_{};
  char* end_{};
};

static_assert(cuda::mr::resource_with<simulated_memory_resource, cuda::mr::device_accessible>);

}  // namespace rmm::mr
