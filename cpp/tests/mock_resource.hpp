/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <gmock/gmock.h>

#include <cstddef>

namespace rmm::test {

class mock_resource {
 public:
  MOCK_METHOD(void*, allocate, (cuda::stream_ref, std::size_t, std::size_t));
  MOCK_METHOD(void, deallocate, (cuda::stream_ref, void*, std::size_t, std::size_t), (noexcept));

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate(rmm::cuda_stream_view{}, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(rmm::cuda_stream_view{}, ptr, bytes, alignment);
  }

  bool operator==(mock_resource const&) const noexcept { return true; }
  bool operator!=(mock_resource const&) const { return false; }
  friend void get_property(mock_resource const&, cuda::mr::device_accessible) noexcept {}
  using size_pair = std::pair<std::size_t, std::size_t>;
};

// static property checks
static_assert(cuda::mr::resource_with<mock_resource, cuda::mr::device_accessible>);

// Copyable wrapper around mock_resource that satisfies CCCL basic_any's requirements.
// GMock types are not copyable, so they cannot be type-erased by CCCL's resource_ref
// (which uses basic_any internally). This thin forwarding layer solves that.
class mock_resource_wrapper {
 public:
  explicit mock_resource_wrapper(mock_resource* mock) noexcept : mock_{mock} {}

  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    return mock_->allocate(stream, bytes, alignment);
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    mock_->deallocate(stream, ptr, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return mock_->allocate_sync(bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    mock_->deallocate_sync(ptr, bytes, alignment);
  }

  bool operator==(mock_resource_wrapper const& other) const noexcept
  {
    return mock_ == other.mock_;
  }
  bool operator!=(mock_resource_wrapper const& other) const noexcept { return !(*this == other); }

  friend void get_property(mock_resource_wrapper const&, cuda::mr::device_accessible) noexcept {}

 private:
  mock_resource* mock_;
};

static_assert(cuda::mr::resource_with<mock_resource_wrapper, cuda::mr::device_accessible>);

}  // namespace rmm::test
