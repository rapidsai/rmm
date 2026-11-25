/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/fixed_slab_memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda/stream>

#include <cstddef>
#include <utility>

RMM_NAMESPACE_BEGIN
namespace mr {

fixed_slab_memory_resource::fixed_slab_memory_resource(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
  cuda::stream_ref stream,
  std::size_t slab_bytes)
  : shared_base(cuda::mr::make_shared_resource<detail::fixed_slab_memory_resource_impl>(
      std::move(upstream), stream, slab_bytes))
{
}

void* fixed_slab_memory_resource::allocate(cuda::stream_ref stream,
                                           std::size_t bytes,
                                           std::size_t alignment)
{
  return get().allocate(stream, bytes, alignment);
}

void fixed_slab_memory_resource::deallocate(cuda::stream_ref stream,
                                            void* ptr,
                                            std::size_t bytes,
                                            std::size_t alignment) noexcept
{
  get().deallocate(stream, ptr, bytes, alignment);
}

void* fixed_slab_memory_resource::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return get().allocate_sync(bytes, alignment);
}

void fixed_slab_memory_resource::deallocate_sync(void* ptr,
                                                 std::size_t bytes,
                                                 std::size_t alignment) noexcept
{
  get().deallocate_sync(ptr, bytes, alignment);
}

std::size_t fixed_slab_memory_resource::slab_bytes() const noexcept { return get().slab_bytes(); }

void* fixed_slab_memory_resource::data() const noexcept { return get().data(); }

std::size_t fixed_slab_memory_resource::carved_bytes() const noexcept
{
  return get().carved_bytes();
}

}  // namespace mr
RMM_NAMESPACE_END
