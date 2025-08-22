
/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

  void* allocate(std::size_t bytes)
  {
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    return base::allocate_sync(bytes);
#else
    return base::allocate(bytes);
#endif
  }

  void* allocate(std::size_t bytes, std::size_t alignment)
  {
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    return base::allocate_sync(bytes, alignment);
#else
    return base::allocate(bytes, alignment);
#endif
  }

  void deallocate(void* ptr, std::size_t bytes)
  {
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    return base::deallocate_sync(ptr, bytes);
#else
    return base::deallocate(ptr, bytes);
#endif
  }

  void deallocate(void* ptr, std::size_t bytes, std::size_t alignment)
  {
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    return base::deallocate_sync(ptr, bytes, alignment);
#else
    return base::deallocate(ptr, bytes, alignment);
#endif
  }
};

template <typename ResourceType>
class cccl_async_resource_ref : public ResourceType {
 public:
  using base = ResourceType;

  using base::base;

  cccl_async_resource_ref(base const& other) : base(other) {}
  cccl_async_resource_ref(base&& other) : base(std::move(other)) {}

  void* allocate(std::size_t bytes)
  {
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    return base::allocate_sync(bytes);
#else
    return base::allocate(bytes);
#endif
  }

  void* allocate(std::size_t bytes, std::size_t alignment)
  {
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    return base::allocate_sync(bytes, alignment);
#else
    return base::allocate(bytes, alignment);
#endif
  }

  void deallocate(void* ptr, std::size_t bytes)
  {
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    return base::deallocate_sync(ptr, bytes);
#else
    return base::deallocate(ptr, bytes);
#endif
  }

  void deallocate(void* ptr, std::size_t bytes, std::size_t alignment)
  {
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    return base::deallocate_sync(ptr, bytes, alignment);
#else
    return base::deallocate(ptr, bytes, alignment);
#endif
  }

  void* allocate_async(std::size_t bytes, cuda_stream_view stream)
  {
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    return base::allocate(stream, bytes);
#else
    return base::allocate_sync(bytes, stream);
#endif
  }

  void* allocate_async(std::size_t bytes, std::size_t alignment, cuda_stream_view stream)
  {
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    return base::allocate(stream, bytes, alignment);
#else
    return base::allocate_sync(bytes, alignment, stream);
#endif
  }

  void deallocate_async(void* ptr, std::size_t bytes, cuda_stream_view stream)
  {
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    return base::deallocate(stream, ptr, bytes);
#else
    return base::deallocate_sync(ptr, bytes, stream);
#endif
  }

  void deallocate_async(void* ptr,
                        std::size_t bytes,
                        std::size_t alignment,
                        cuda_stream_view stream)
  {
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
    return base::deallocate(stream, ptr, bytes, alignment);
#else
    return base::deallocate_sync(ptr, bytes, alignment, stream);
#endif
  }
};

#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
#define RMM_CCCL_MR_METHODS                                                 \
 public:                                                                    \
  void* allocate_sync(std::size_t bytes, std::size_t alignment)             \
  {                                                                         \
    return this->allocate(bytes, alignment);                                \
  }                                                                         \
                                                                            \
  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) \
  {                                                                         \
    return this->deallocate(ptr, bytes, alignment);                         \
  }
#else
#define RMM_CCCL_MR_METHODS
#endif

#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1)
#define RMM_CCCL_ASYNC_MR_METHODS                                                               \
  RMM_CCCL_MR_METHODS                                                                           \
                                                                                                \
 public:                                                                                        \
  void* allocate(cuda_stream_view stream, std::size_t bytes, std::size_t alignment)             \
  {                                                                                             \
    return this->allocate_async(bytes, alignment, stream);                                      \
  }                                                                                             \
                                                                                                \
  void deallocate(cuda_stream_view stream, void* ptr, std::size_t bytes, std::size_t alignment) \
  {                                                                                             \
    return this->deallocate_async(ptr, bytes, alignment, stream);                               \
  }
#else
#define RMM_CCCL_ASYNC_MR_METHODS
#endif

}  // namespace detail
}  // namespace RMM_NAMESPACE
