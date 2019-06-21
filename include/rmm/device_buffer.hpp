/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <rmm/mr/default_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cuda_runtime_api.h>
#include <cassert>

namespace rmm {
/**---------------------------------------------------------------------------*
 * @file device_buffer.hpp
 * @brief RAII construct for device memory allocation
 *---------------------------------------------------------------------------**/

class device_buffer {
 public:
  device_buffer() = delete;

  /**---------------------------------------------------------------------------*
   * @brief Constructs a new device buffer of `size` unitialized bytes
   *
   * @param size Size in bytes to allocate in device memory
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams, else the null stream is used.
   * @param mr Memory resource to use for the device memory allocation
   *---------------------------------------------------------------------------**/
  explicit device_buffer(
      std::size_t size, cudaStream_t stream = 0,
      mr::device_memory_resource* mr = mr::get_default_resource())
      : _size{size}, _stream{stream}, _mr{mr} {
    _data = _mr->allocate(size, stream);
  }

  /**---------------------------------------------------------------------------*
   * @brief Constructs a new `device_buffer` by deep copying the contents of
   * another `device_buffer`.
   *
   * @param other The other `device_buffer` to deep copy
   *---------------------------------------------------------------------------**/
  device_buffer(device_buffer const& other)
      : _size{other._size}, _stream{other._stream}, _mr{other._mr} {
    _data = _mr->allocate(_size, _stream);
    auto status =
        cudaMemcpyAsync(_data, other._data, _size, cudaMemcpyDefault, _stream);

    if (cudaSuccess != status) {
      throw std::runtime_error{"Device memory copy failed."};
    }
  }

  /**---------------------------------------------------------------------------*
   * @brief Constructs a new `device_buffer` by moving the contents of another
   * `device_buffer` into the newly constructed one.
   *
   * After the new `device_buffer` is constructed, `other` is modified to be
   * empty, i.e., `data()` returns `nullptr`, and `size()` is zero.
   *
   * @param other The `device_buffer` whose contents will be moved into the
   * newly constructed one
   *---------------------------------------------------------------------------**/
  device_buffer(device_buffer&& other)
      : _data{other._data},
        _size{other._size},
        _stream{other._stream},
        _mr{other._mr} {
    other._data = nullptr;
    other._size = 0;
    other._stream = 0;
  }

  /**---------------------------------------------------------------------------*
   * @brief Copy assignment operator
   *
   * TODO: Decide if this should be deleted or not.
   *
   * @param other
   * @return device_buffer&
   *---------------------------------------------------------------------------**/
  device_buffer& operator=(device_buffer const& other) =
      delete
      /*
      {
        if (&other != this) {
          _mr->deallocate(_data, _size, _stream);
          _size = other._size;
          _stream = other._stream;
          _mr = other._mr;
          _data = _mr->allocate(_size, _stream);
          auto status = cudaMemcpyAsync(_data, other._data, _size,
                                        cudaMemcpyDefault, _stream);

          if (cudaSuccess != status) {
            throw std::runtime_error{"Device memory copy failed."};
          }
        }
        return *this;
      }
      */

      /**---------------------------------------------------------------------------*
       * @brief Move assignment operator
       *
       * TODO: Decide if this should be deleted or not.
       *
       * @param other
       * @return device_buffer&
       *---------------------------------------------------------------------------**/
      device_buffer
      & operator=(device_buffer&& other) = delete;
  /*
   {
    if (&other != this) {
      _mr->deallocate(_data, _size, _stream);
      _data = other._data;
      _size = other._size;
      _stream = other._stream;
      _mr = other._mr;

      other._data = nullptr;
      other._size = 0;
      other._stream = 0;
    }
    return *this;
  }
  */

  ~device_buffer() {
    _mr->deallocate(_data, _size, _stream);
    _data = nullptr;
    _size = 0;
    _stream = 0;
    _mr = nullptr;
  }

  /**---------------------------------------------------------------------------*
   * @brief Resize the device memory allocation
   *
   * If the requested `new_size` is less than or equal to the current size, no
   * action is taken other than updating the value that is returned from
   * `size()`. I.e., no memory is allocated nor copied.
   *
   * If `new_size` is larger than the current size, a new allocation is made to
   * satisfy `new_size`, and the contents of the old allocation are copied to
   * the new allocation. The old allocation is then freed.
   *
   * The `stream` returned by `stream()` is used for the allocation and copying
   * of the new memory.
   *
   * @param new_size The requested new size, in bytes
   *---------------------------------------------------------------------------**/
  void resize(std::size_t new_size) {
    // If the requested size is smaller, just update the size without any
    // allocations
    if (new_size <= _size) {
      _size = new_size;
    } else {
      void* const new_data = _mr->allocate(new_size, _stream);
      if (_size > 0) {
        auto status =
            cudaMemcpyAsync(new_data, _data, _size, cudaMemcpyDefault, _stream);

        if (cudaSuccess != status) {
          throw std::runtime_error{"Device memory copy failed."};
        }
      }
      _mr->deallocate(_data, _size, _stream);
      _data = new_data;
      _size = new_size;
    }
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns raw pointer to underlying device memory allocation
   *---------------------------------------------------------------------------**/
  void const* data() const { return _data; }

  /**---------------------------------------------------------------------------*
   * @brief Returns raw pointer to underlying device memory allocation
   *---------------------------------------------------------------------------**/
  void* data() { return _data; }

  /**---------------------------------------------------------------------------*
   * @brief Returns size in bytes of device memory allocation
   *---------------------------------------------------------------------------**/
  std::size_t size() const { return _size; }

  /**---------------------------------------------------------------------------*
   * @brief Returns stream used for allocation/deallocation
   *---------------------------------------------------------------------------**/
  cudaStream_t stream() const { return _stream; }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to the memory resource used to allocate and
   * deallocate the device memory
   *---------------------------------------------------------------------------**/
  mr::device_memory_resource* memory_resource() const { return _mr; }

 private:
  void* _data{nullptr};     ///< Pointer to device memory allocation
  std::size_t _size{0};     ///< Size in bytes of device memory allocation
  cudaStream_t _stream{0};  ///< Stream which may be used for
                            ///< allocation/deallocation of device memory
  mr::device_memory_resource* _mr{
      mr::get_default_resource()};  ///< The memory resource used to
                                    ///< allocate/deallocate device memory
};
}  // namespace rmm