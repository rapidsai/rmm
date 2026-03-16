/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <chrono>
#include <cstddef>
#include <memory>
#include <thread>

namespace rmm::test {

/**
 * @brief A memory resource that wraps an upstream and adds a delay after deallocation.
 *
 * This is useful for testing ABA problems in resource adaptors. The delay simulates the window
 * where the upstream has freed a pointer (making the address available for reuse) but the calling
 * thread has not yet returned to update its bookkeeping.
 */
class delayed_memory_resource {
 public:
  delayed_memory_resource(rmm::device_async_resource_ref upstream, std::chrono::milliseconds delay)
    : upstream_{upstream}, delay_{delay}
  {
  }
  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    return upstream_.allocate_sync(bytes, alignment);
  }
  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment)
  {
    upstream_.deallocate_sync(ptr, bytes, alignment);
    std::this_thread::sleep_for(delay_);
  }
  void* allocate(rmm::cuda_stream_view stream, std::size_t bytes, std::size_t alignment)
  {
    return upstream_.allocate(stream, bytes, alignment);
  }
  void deallocate(rmm::cuda_stream_view stream, void* ptr, std::size_t bytes, std::size_t alignment)
  {
    upstream_.deallocate(stream, ptr, bytes, alignment);
    std::this_thread::sleep_for(delay_);
  }
  friend void get_property(delayed_memory_resource const&, cuda::mr::device_accessible) noexcept {}
  bool operator==(delayed_memory_resource const& other) const noexcept
  {
    return this == std::addressof(other);
  }

  bool operator!=(delayed_memory_resource const& other) const noexcept
  {
    return !(this == std::addressof(other));
  }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
  std::chrono::milliseconds delay_;
};
static_assert(cuda::mr::resource<delayed_memory_resource>);
static_assert(cuda::mr::resource_with<delayed_memory_resource, cuda::mr::device_accessible>);

}  // namespace rmm::test
