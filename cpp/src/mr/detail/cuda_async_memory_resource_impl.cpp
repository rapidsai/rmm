/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/mr/detail/cuda_async_memory_resource_impl.hpp>
#include <rmm/process_is_exiting.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <driver_types.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>

RMM_NAMESPACE_BEGIN
namespace mr {
namespace detail {
namespace {

struct retained_pool_state {
  std::mutex lock;
  cudaMemPool_t pool{};
};

retained_pool_state& get_retained_pool_state(rmm::cuda_device_id device_id)
{
  static std::mutex lock;
  static std::map<rmm::cuda_device_id::value_type, retained_pool_state> states;
  std::lock_guard guard{lock};
  return states.try_emplace(device_id.value()).first->second;
}

cudaMemPool_t create_pool(cudaMemAllocationHandleType handle_type,
                          bool enable_hw_decompress,
                          std::uint64_t release_threshold)
{
  RMM_EXPECTS(rmm::detail::export_handle_type::is_supported(handle_type),
              "Requested IPC memory handle type not supported");

  // Construct explicit pool
  cudaMemPoolProps pool_props{};
  pool_props.allocType     = cudaMemAllocationTypePinned;
  pool_props.handleTypes   = handle_type;
  pool_props.location.type = cudaMemLocationTypeDevice;
  pool_props.location.id   = rmm::get_current_cuda_device().value();

#if CUDART_VERSION >= RMM_MIN_HWDECOMPRESS_CUDA_VERSION
  // usage field in the cudaMemPoolProps only exists in new enough versions of the runtime
  // headers.
  if (enable_hw_decompress) { pool_props.usage = cudaMemPoolCreateUsageHwDecompress; }
#else
  (void)enable_hw_decompress;
#endif

  cudaMemPool_t pool{};
  RMM_CUDA_TRY(cudaMemPoolCreate(&pool, &pool_props));
  RMM_CUDA_TRY(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &release_threshold));
  return pool;
}

cudaMemPool_t get_current_pool(bool enable_hw_decompress)
{
  auto const device_id = rmm::get_current_cuda_device();
  auto& state          = get_retained_pool_state(device_id);
  std::lock_guard guard{state.lock};

  cudaMemPool_t current_pool{};
  cudaMemPool_t default_pool{};
  RMM_CUDA_TRY(cudaDeviceGetMemPool(&current_pool, device_id.value()));
  RMM_CUDA_TRY(cudaDeviceGetDefaultMemPool(&default_pool, device_id.value()));
  if (current_pool != default_pool) { return current_pool; }

  auto constexpr max_threshold = std::numeric_limits<std::uint64_t>::max();
  if (!enable_hw_decompress) {
    // Need an l-value to take address to pass to cudaMemPoolSetAttribute
    auto threshold = max_threshold;
    RMM_CUDA_TRY(
      cudaMemPoolSetAttribute(default_pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    return default_pool;
  }

  bool created{};
  if (state.pool == nullptr) {
    state.pool = create_pool(cudaMemHandleTypeNone, true, max_threshold);
    created    = true;
  }
  try {
    RMM_CUDA_TRY(cudaDeviceSetMemPool(device_id.value(), state.pool));
  } catch (...) {
    if (created) {
      cudaMemPoolDestroy(state.pool);
      state.pool = nullptr;
    }
    throw;
  }
  return state.pool;
}

}  // namespace

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
cuda_async_memory_resource_impl::cuda_async_memory_resource_impl(
  std::optional<std::size_t> initial_pool_size,
  std::optional<std::size_t> release_threshold,
  std::optional<std::int32_t> export_handle_type,
  bool enable_hw_decompress)
{
  RMM_EXPECTS(rmm::detail::runtime_async_alloc::is_supported(),
              "cudaMallocAsync not supported with this CUDA driver/runtime version");

  auto const handle_type =
    static_cast<cudaMemAllocationHandleType>(export_handle_type.value_or(cudaMemHandleTypeNone));
  owns_pool_ = release_threshold.has_value() || handle_type != cudaMemHandleTypeNone;
  if (owns_pool_) {
    auto const threshold = release_threshold.value_or(0) == 0
                             ? std::numeric_limits<std::uint64_t>::max()
                             : release_threshold.value();
    pool_ =
      cuda_async_view_memory_resource{create_pool(handle_type, enable_hw_decompress, threshold)};
  } else {
    pool_ = cuda_async_view_memory_resource{get_current_pool(enable_hw_decompress)};
  }

  // Allocate and immediately deallocate the initial_pool_size to prime the pool with the
  // specified size (only if initial_pool_size is provided)
  if (initial_pool_size.has_value()) {
    auto const pool_size = initial_pool_size.value();
    auto* ptr            = allocate(cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, pool_size);
    deallocate(cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, ptr, pool_size);
  }
}

cuda_async_memory_resource_impl::~cuda_async_memory_resource_impl()
{
  if (!owns_pool_ || rmm::process_is_exiting()) { return; }

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
  auto* ptr = allocate(cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, bytes, alignment);
  RMM_CUDA_TRY(cudaStreamSynchronize(cudaStream_t{nullptr}));
  return ptr;
}

void cuda_async_memory_resource_impl::deallocate_sync(void* ptr,
                                                      std::size_t bytes,
                                                      std::size_t alignment) noexcept
{
  auto const stream = cuda::stream_ref{cudaStream_t{cudaStreamDefault}};
  deallocate(stream, ptr, bytes, alignment);
  RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaStreamSynchronize(stream.get()));
}

}  // namespace detail
}  // namespace mr
RMM_NAMESPACE_END
