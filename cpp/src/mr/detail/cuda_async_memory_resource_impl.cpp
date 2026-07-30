/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/mr/detail/cuda_async_memory_resource_impl.hpp>
#include <rmm/process_is_exiting.hpp>

#include <cuda_runtime_api.h>

#include <driver_types.h>

#include <cstddef>
#include <cstdint>
#include <limits>

RMM_NAMESPACE_BEGIN
namespace mr {
namespace detail {

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
cuda_async_memory_resource_impl::cuda_async_memory_resource_impl(
  std::optional<std::size_t> initial_pool_size,
  std::optional<std::size_t> release_threshold,
  std::optional<std::int32_t> export_handle_type,
  bool enable_hw_decompress)
{
  RMM_EXPECTS(rmm::detail::runtime_async_alloc::is_supported(),
              "cudaMallocAsync not supported with this CUDA driver/runtime version");

  // Construct explicit pool
  cudaMemPoolProps pool_props{};
  pool_props.allocType = cudaMemAllocationTypePinned;
  pool_props.handleTypes =
    static_cast<cudaMemAllocationHandleType>(export_handle_type.value_or(cudaMemHandleTypeNone));

#if CUDART_VERSION >= RMM_MIN_HWDECOMPRESS_CUDA_VERSION
  // usage field in the cudaMemPoolProps only exists in new enough versions of the runtime
  // headers.
  if (enable_hw_decompress) { pool_props.usage = cudaMemPoolCreateUsageHwDecompress; }
#else
  (void)enable_hw_decompress;
#endif

  RMM_EXPECTS(rmm::detail::export_handle_type::is_supported(pool_props.handleTypes),
              "Requested IPC memory handle type not supported");
  pool_props.location.type = cudaMemLocationTypeDevice;
  pool_props.location.id   = rmm::get_current_cuda_device().value();
  cudaMemPool_t cuda_pool_handle{};
  RMM_CUDA_TRY(cudaMemPoolCreate(&cuda_pool_handle, &pool_props));
  pool_ = cuda_async_view_memory_resource{cuda_pool_handle};

  // Need an l-value to take address to pass to cudaMemPoolSetAttribute
  std::uint64_t threshold = release_threshold.value_or(std::numeric_limits<std::uint64_t>::max());
  RMM_CUDA_TRY(cudaMemPoolSetAttribute(pool_handle(), cudaMemPoolAttrReleaseThreshold, &threshold));

  // Allocate and immediately deallocate the initial_pool_size to prime the pool with the
  // specified size (only if initial_pool_size is provided)
  if (initial_pool_size.has_value()) {
    auto const pool_size = initial_pool_size.value();
    auto* ptr            = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, pool_size);
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, pool_size);
  }
}

cuda_async_memory_resource_impl::~cuda_async_memory_resource_impl()
{
  if (rmm::process_is_exiting()) { return; }

  RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaMemPoolDestroy(pool_handle()));
}

cudaMemPool_t cuda_async_memory_resource_impl::pool_handle() const noexcept
{
  return pool_.pool_handle();
}

void* cuda_async_memory_resource_impl::allocate(cuda::stream_ref stream,
                                                std::size_t bytes,
                                                std::size_t alignment)
{
  return pool_.allocate(stream, bytes, alignment);
}

void cuda_async_memory_resource_impl::deallocate(cuda::stream_ref stream,
                                                 void* ptr,
                                                 std::size_t bytes,
                                                 std::size_t /*alignment*/) noexcept
{
  pool_.deallocate(stream, ptr, bytes);
}

void* cuda_async_memory_resource_impl::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  auto* ptr = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  RMM_CUDA_TRY(cudaStreamSynchronize(cudaStream_t{nullptr}));
  return ptr;
}

void cuda_async_memory_resource_impl::deallocate_sync(void* ptr,
                                                      std::size_t bytes,
                                                      std::size_t alignment) noexcept
{
  auto const stream = cuda::stream_ref{cudaStream_t{nullptr}};
  deallocate(stream, ptr, bytes, alignment);
  RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaStreamSynchronize(stream.get()));
}

}  // namespace detail
}  // namespace mr
RMM_NAMESPACE_END
