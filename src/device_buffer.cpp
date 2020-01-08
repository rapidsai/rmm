/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <rmm/device_buffer.hpp>

namespace rmm {
void copy_to_host(const device_buffer& db, void* hb, cudaStream_t stream) {
  if (hb == nullptr) {
    throw std::runtime_error{"Cannot copy to `nullptr`."};
  }
  cudaError_t err = cudaMemcpyAsync(hb,
                                    db.data(),
                                    db.size(),
                                    cudaMemcpyDeviceToHost,
                                    stream);
  if (err != cudaSuccess) {
    throw std::runtime_error{"Failed to copy to host."};
  }
}
}  // namespace rmm
