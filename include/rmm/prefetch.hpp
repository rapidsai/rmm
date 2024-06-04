/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/error.hpp>

#include <iterator>

namespace rmm {

/**
 * @brief Prefetch data for this buffer to the specified device on the specified stream.
 *
 * This function is a no-op if the pointer is not to CUDA managed memory.
 *
 * @throw rmm::cuda_error if the prefetch fails.
 *
 * @tparam T The type of the elements pointed to by `ptr`.
 * @param ptr The pointer to the memory to prefetch
 * @param size The number of bytes to prefetch
 * @param device The device to prefetch to
 * @param stream The stream to use for the prefetch
 */
template <typename T>
void prefetch(T* ptr, std::size_t size, rmm::cuda_device_id device, rmm::cuda_stream_view stream)
{
  auto result = cudaMemPrefetchAsync(ptr, size, device.value(), stream.value());
  // InvalidValue error is raised when non-managed memory is passed to cudaMemPrefetchAsync
  // We should treat this as a no-op
  if (result != cudaErrorInvalidValue && result != cudaSuccess) { RMM_CUDA_TRY(result); }
}

/**
 * @brief Prefetch a range of memory to the specified device on the specified stream.
 *
 * This function is a no-op if the Iterators do not reference CUDA managed memory.
 *
 * @throw rmm::cuda_error if the prefetch fails.
 *
 * @tparam Iterator The type of the iterator parameters
 * @param begin The beginning of the range to prefetch
 * @param end The end of the range to prefetch
 * @param device The device to prefetch to
 * @param stream The stream to use for the prefetch
 */
template <typename Iterator>
void prefetch(Iterator begin,
              Iterator end,
              rmm::cuda_device_id device,
              rmm::cuda_stream_view stream)
{
  constexpr auto elt_size = sizeof(typename std::iterator_traits<Iterator>::value_type);
  prefetch(begin, std::distance(begin, end) * elt_size, device, stream);
}

/**
 * @brief Prefetch the storage for a container to the specified device on the specified stream.
 *
 * This function is a no-op if the container is not backed by CUDA managed memory.
 * This function requires the container to implement `begin()` and `end()` which return iterators.
 *
 * @throw rmm::cuda_error if the prefetch fails.
 *
 * @tparam Container The type of the container parameter
 * @param container The container to prefetch
 * @param device The device to prefetch to
 * @param stream The stream to use for the prefetch
 */
template <typename Container>
void prefetch(Container const& container, rmm::cuda_device_id device, rmm::cuda_stream_view stream)
{
  prefetch(container.begin(), container.end(), device, stream);
}

/**
    @brief Prefetch the storage for a rmm::device_buffer to the specified device on the specified
   stream.

    This function is a no-op if the buffer is not backed by CUDA managed memory.

    @throw rmm::cuda_error if the prefetch fails.

    @param buffer The buffer to prefetch
    @param device The device to prefetch to
    @param stream The stream to use for the prefetch
 */
void prefetch(rmm::device_buffer const& buffer,
              rmm::cuda_device_id device,
              rmm::cuda_stream_view stream)
{
  prefetch(buffer.data(), buffer.size(), device, stream);
}

/**
    @brief Prefetch the storage for a rmm::device_scalar to the specified device on the specified
   stream.

    This function is a no-op if the buffer is not backed by CUDA managed memory.

    @throw rmm::cuda_error if the prefetch fails.

    @param scalar The device_scalar to prefetch
    @param device The device to prefetch to
    @param stream The stream to use for the prefetch
 */
template <typename T>
void prefetch(rmm::device_scalar<T> const& scalar,
              rmm::cuda_device_id device,
              rmm::cuda_stream_view stream)
{
  prefetch(scalar.data(), sizeof(T), device, stream);
}

}  // namespace rmm
