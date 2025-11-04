/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/cuda_memory_resource.hpp>
#include <rmm/detail/export.hpp>

namespace RMM_NAMESPACE {

namespace detail {

template <typename ResourceType>
class cccl_resource_ref : public ResourceType {
 public:
  using base = ResourceType;

  using base::base;

  cccl_resource_ref(base const& other) : base(other) {}

  cccl_resource_ref(base&& other) : base(std::move(other)) {}

#ifdef RMM_ENABLE_LEGACY_MR_INTERFACE
  void* allocate(std::size_t bytes) { return this->allocate_sync(bytes); }

  void* allocate(std::size_t bytes, std::size_t alignment)
  {
    return this->allocate_sync(bytes, alignment);
  }

  void deallocate(void* ptr, std::size_t bytes) noexcept
  {
    return this->deallocate_sync(ptr, bytes);
  }

  void deallocate(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    return this->deallocate_sync(ptr, bytes, alignment);
  }
#endif  // RMM_ENABLE_LEGACY_MR_INTERFACE

  void* allocate_sync(std::size_t bytes) { return base::allocate_sync(bytes); }

  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    return base::allocate_sync(bytes, alignment);
  }

  void deallocate_sync(void* ptr, std::size_t bytes) noexcept
  {
    return base::deallocate_sync(ptr, bytes);
  }

  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    return base::deallocate_sync(ptr, bytes, alignment);
  }
};

template <typename ResourceType>
class cccl_async_resource_ref : public ResourceType {
 public:
  using base = ResourceType;

  using base::base;

  cccl_async_resource_ref(base const& other) : base(other) {}
  cccl_async_resource_ref(base&& other) : base(std::move(other)) {}

#ifdef RMM_ENABLE_LEGACY_MR_INTERFACE
  void* allocate(std::size_t bytes) { return this->allocate_sync(bytes); }

  void* allocate(std::size_t bytes, std::size_t alignment)
  {
    return this->allocate_sync(bytes, alignment);
  }

  void deallocate(void* ptr, std::size_t bytes) noexcept
  {
    return this->deallocate_sync(ptr, bytes);
  }

  void deallocate(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    return this->deallocate_sync(ptr, bytes, alignment);
  }

  void* allocate_async(std::size_t bytes, cuda_stream_view stream)
  {
    return this->allocate(stream, bytes);
  }

  void* allocate_async(std::size_t bytes, std::size_t alignment, cuda_stream_view stream)
  {
    return this->allocate(stream, bytes, alignment);
  }

  void deallocate_async(void* ptr, std::size_t bytes, cuda_stream_view stream) noexcept
  {
    return this->deallocate(stream, ptr, bytes);
  }

  void deallocate_async(void* ptr,
                        std::size_t bytes,
                        std::size_t alignment,
                        cuda_stream_view stream) noexcept
  {
    return this->deallocate(stream, ptr, bytes, alignment);
  }

#endif  // RMM_ENABLE_LEGACY_MR_INTERFACE

  void* allocate_sync(std::size_t bytes) { return base::allocate_sync(bytes); }

  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    return base::allocate_sync(bytes, alignment);
  }

  void deallocate_sync(void* ptr, std::size_t bytes) noexcept
  {
    return base::deallocate_sync(ptr, bytes);
  }

  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    return base::deallocate_sync(ptr, bytes, alignment);
  }

  void* allocate(cuda_stream_view stream, std::size_t bytes)
  {
    return base::allocate(stream, bytes);
  }

  void* allocate(cuda_stream_view stream, std::size_t bytes, std::size_t alignment)
  {
    return base::allocate(stream, bytes, alignment);
  }

  void deallocate(cuda_stream_view stream, void* ptr, std::size_t bytes) noexcept
  {
    return base::deallocate(stream, ptr, bytes);
  }

  void deallocate(cuda_stream_view stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    return base::deallocate(stream, ptr, bytes, alignment);
  }
};

}  // namespace detail
}  // namespace RMM_NAMESPACE
