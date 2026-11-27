/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/detail/fixed_slab_memory_resource_impl.hpp>
#include <rmm/process_is_exiting.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

RMM_NAMESPACE_BEGIN
namespace mr {
namespace detail {

fixed_slab_memory_resource_impl::fixed_slab_memory_resource_impl(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
  cuda::stream_ref stream,
  std::size_t slab_bytes)
  : upstream_mr_{std::move(upstream)},
    slab_bytes_{slab_bytes},
    construction_stream_{normalize_stream(stream)}
{
  RMM_EXPECTS(slab_bytes != 0, "Slab size must be greater than zero.");
  RMM_EXPECTS(rmm::is_aligned(slab_bytes, rmm::CUDA_ALLOCATION_ALIGNMENT),
              "Slab size required to be a multiple of the CUDA allocation alignment");

  slab_ = upstream_mr_.allocate(
    cuda::stream_ref{construction_stream_}, slab_bytes_, rmm::CUDA_ALLOCATION_ALIGNMENT);
  RMM_EXPECTS(slab_ != nullptr, "Upstream resource returned a null slab.", rmm::out_of_memory);
}

fixed_slab_memory_resource_impl::~fixed_slab_memory_resource_impl() noexcept { release(); }

void fixed_slab_memory_resource_impl::check_stream(cudaStream_t stream) const
{
  RMM_EXPECTS(stream == construction_stream_,
              "fixed_slab_memory_resource is single-stream: the slab lives on the stream it "
              "was allocated on, and all allocate/deallocate calls must use that stream.",
              rmm::logic_error);
}

std::size_t fixed_slab_memory_resource_impl::carve_offset(std::size_t bytes,
                                                          std::size_t alignment) const
{
  auto const effective = std::max(alignment, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto const remaining = slab_bytes_ - offset_;

  if (bytes > remaining) {
    throw rmm::out_of_memory{std::string{"Failed to allocate "} + std::to_string(bytes) +
                             " bytes from fixed slab with " + std::to_string(remaining) +
                             " bytes remaining"};
  }

  // Compute the padding needed to align the actual address `slab_ + offset_`, not just the
  // offset: a base aligned to the allocation alignment but not to a larger alignment must
  // still produce a correctly aligned result. Both masked operands are at most `effective - 1`,
  // so the intermediate
  // sum is in range for every representable power-of-two alignment.
  auto const mask    = effective - 1;
  auto const base    = reinterpret_cast<std::uintptr_t>(slab_);
  auto const low     = ((base & mask) + (offset_ & mask)) & mask;
  auto const padding = (effective - low) & mask;

  if (padding > remaining - bytes) {
    throw rmm::out_of_memory{std::string{"Failed to allocate "} + std::to_string(bytes) +
                             " bytes aligned to " + std::to_string(effective) +
                             " bytes from fixed slab with " + std::to_string(remaining) +
                             " bytes remaining"};
  }

  return offset_ + padding;
}

void* fixed_slab_memory_resource_impl::allocate(cuda::stream_ref stream,
                                                std::size_t bytes,
                                                std::size_t alignment)
{
  // Stream validation comes FIRST: even a zero-byte carve must reject a wrong stream, so
  // the mismatch never slips through a size edge case.
  check_stream(normalize_stream(stream));
  // Every carve is pure host arithmetic ordered behind the slab allocation on the recorded
  // stream. The carve issues no stream work — ordering comes from check_stream and the
  // single-stream slab contract — so delegating to the sync body cannot disturb the recorded
  // stream.
  return allocate_sync(bytes, alignment);
}

void fixed_slab_memory_resource_impl::deallocate(cuda::stream_ref stream,
                                                 void* ptr,
                                                 std::size_t bytes,
                                                 std::size_t alignment) noexcept
{
  // Documented contract: null/zero calls are ignored WITHOUT validation.
  if (bytes == 0 || ptr == nullptr) { return; }
  // noexcept path: cannot throw rmm::logic_error, so the mismatch fails closed directly
  // (branch-only, zero allocation) — the same contract the throwing paths enforce.
  if (normalize_stream(stream) != construction_stream_) { std::terminate(); }
  // No stream work to undo, so the body is exactly the sync no-op.
  deallocate_sync(ptr, bytes, alignment);
}

void* fixed_slab_memory_resource_impl::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  if (bytes == 0) { return nullptr; }

  RMM_EXPECTS(rmm::is_supported_alignment(alignment), "Allocation alignment is not a power of 2.");

  // Host-only carve per design: NO wait is performed. The caller must have synchronized the
  // recorded stream before host-synchronous use of the slice.
  auto const start = carve_offset(bytes, alignment);
  auto const end   = start + bytes;
  offset_          = end;
  return static_cast<std::byte*>(slab_) + start;
}

void fixed_slab_memory_resource_impl::deallocate_sync(void*, std::size_t, std::size_t) noexcept
{
  // Neither sync entry point issues stream work, so no stream argument could mismatch.
}

void fixed_slab_memory_resource_impl::release() noexcept
{
  if (rmm::process_is_exiting()) { return; }

  // Stream-ordered free on the recorded stream. It is NOT necessarily complete on return;
  // the upstream resource owns completion. The contract is only that no work outside the
  // recorded stream can touch the freed region afterwards, which the single-stream rule
  // guarantees.
  upstream_mr_.deallocate(
    cuda::stream_ref{construction_stream_}, slab_, slab_bytes_, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

}  // namespace detail
}  // namespace mr
RMM_NAMESPACE_END
