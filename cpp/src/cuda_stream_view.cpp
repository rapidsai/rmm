/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
