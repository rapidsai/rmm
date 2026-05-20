/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <utility>

class device_check_resource_adaptor final {
 public:
  device_check_resource_adaptor(rmm::device_async_resource_ref upstream)
    : device_id{rmm::get_current_cuda_device()}, upstream_(upstream)
  {
  }

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept
  {
    return rmm::device_async_resource_ref{upstream_};
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    bool const is_correct_device = check_device_id();
    EXPECT_TRUE(is_correct_device);
    if (is_correct_device) { return get_upstream_resource().allocate(stream, bytes, alignment); }
    return nullptr;
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    bool const is_correct_device = check_device_id();
    EXPECT_TRUE(is_correct_device);
    if (is_correct_device) { get_upstream_resource().deallocate(stream, ptr, bytes, alignment); }
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    auto stream = cuda::stream_ref{cudaStream_t{nullptr}};
    auto* ptr   = allocate(stream, bytes, alignment);
    RMM_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    return ptr;
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  bool operator==(device_check_resource_adaptor const& other) const noexcept
  {
    return get_upstream_resource() == other.get_upstream_resource();
  }

  bool operator!=(device_check_resource_adaptor const& other) const noexcept
  {
    return !(*this == other);
  }

  constexpr friend void get_property(device_check_resource_adaptor const&,
                                     cuda::mr::device_accessible) noexcept
  {
  }

 private:
  [[nodiscard]] bool check_device_id() const { return device_id == rmm::get_current_cuda_device(); }

  rmm::cuda_device_id device_id;
  mutable cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
};

static_assert(cuda::mr::resource_with<device_check_resource_adaptor, cuda::mr::device_accessible>);
