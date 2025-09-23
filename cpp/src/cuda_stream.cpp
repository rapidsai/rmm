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

#include <rmm/cuda_stream.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/logging_assert.hpp>

#include <cuda_runtime_api.h>

#include <type_traits>

namespace rmm {

cuda_stream::cuda_stream(cuda_stream::flags flags)
  : stream_{[flags]() {
              auto* stream = new cudaStream_t;  // NOLINT(cppcoreguidelines-owning-memory)
              // TODO: use std::to_underlying once C++23 is allowed.
              RMM_CUDA_TRY(cudaStreamCreateWithFlags(
                stream, static_cast<std::underlying_type_t<cuda_stream::flags>>(flags)));
              return stream;
            }(),
            [](cudaStream_t* stream) {
              RMM_ASSERT_CUDA_SUCCESS(cudaStreamDestroy(*stream));
              delete stream;  // NOLINT(cppcoreguidelines-owning-memory)
            }}
{
}

bool cuda_stream::is_valid() const { return stream_ != nullptr; }

cudaStream_t cuda_stream::value() const
{
  RMM_LOGGING_ASSERT(is_valid());
  return *stream_;
}

cuda_stream::operator cudaStream_t() const noexcept { return value(); }

cuda_stream_view cuda_stream::view() const { return cuda_stream_view{value()}; }

cuda_stream::operator cuda_stream_view() const { return view(); }

void cuda_stream::synchronize() const { RMM_CUDA_TRY(cudaStreamSynchronize(value())); }

void cuda_stream::synchronize_no_throw() const noexcept
{
  RMM_ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(value()));
}

}  // namespace rmm
