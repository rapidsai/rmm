/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <ostream>

namespace rmm {

cuda_stream_view::cuda_stream_view(cudaStream_t stream) noexcept : stream_{stream} {}

cuda_stream_view::cuda_stream_view(cuda::stream_ref stream) noexcept : stream_{stream.get()} {}

cudaStream_t cuda_stream_view::value() const noexcept { return stream_; }

cuda_stream_view::operator cudaStream_t() const noexcept { return value(); }

cuda_stream_view::operator cuda::stream_ref() const noexcept { return value(); }

bool cuda_stream_view::is_per_thread_default() const noexcept
{
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  return *this == cuda_stream_per_thread || value() == nullptr;
#else
  return *this == cuda_stream_per_thread;
#endif
}

bool cuda_stream_view::is_default() const noexcept
{
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  return *this == cuda_stream_legacy;
#else
  return *this == cuda_stream_legacy || value() == nullptr;
#endif
}

void cuda_stream_view::synchronize() const { RMM_CUDA_TRY(cudaStreamSynchronize(stream_)); }

void cuda_stream_view::synchronize_no_throw() const noexcept
{
  RMM_ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream_));
}

bool operator==(cuda_stream_view lhs, cuda_stream_view rhs) { return lhs.value() == rhs.value(); }

bool operator!=(cuda_stream_view lhs, cuda_stream_view rhs) { return not(lhs == rhs); }

std::ostream& operator<<(std::ostream& os, cuda_stream_view stream)
{
  os << stream.value();
  return os;
}

}  // namespace rmm
