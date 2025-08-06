/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cstring>

namespace rmm {

device_buffer::device_buffer() : _mr{rmm::mr::get_current_device_resource_ref()} {}

device_buffer::device_buffer(std::size_t size,
                             cuda_stream_view stream,
                             device_async_resource_ref mr)
  : _stream{stream}, _mr{mr}
{
  cuda_set_device_raii dev{_device};
  allocate_async(size);
}

device_buffer::device_buffer(std::size_t size,
                             cuda_stream_view stream,
                             memory_resource_args const& mr_args)
  : _stream{stream}, _mr{mr_args.device_mr}, _host_mr{mr_args.bounce_buffer_host_mr}
{
  cuda_set_device_raii dev{_device};
  allocate_async(size);
}

device_buffer::device_buffer(void const* source_data,
                             std::size_t size,
                             cuda_stream_view stream,
                             device_async_resource_ref mr)
  : _stream{stream}, _mr{mr}
{
  cuda_set_device_raii dev{_device};
  allocate_async(size);
  copy_async(source_data, size);
}

device_buffer::device_buffer(void const* source_data,
                             std::size_t size,
                             cuda_stream_view stream,
                             memory_resource_args const& mr_args)
  : _stream{stream}, _mr{mr_args.device_mr}, _host_mr{mr_args.bounce_buffer_host_mr}
{
  cuda_set_device_raii dev{_device};
  allocate_async(size);
  copy_async(source_data, size);
}

device_buffer::device_buffer(device_buffer const& other,
                             cuda_stream_view stream,
                             device_async_resource_ref mr)
  : device_buffer{other.data(), other.size(), stream, mr}
{
}

device_buffer::device_buffer(device_buffer const& other,
                             cuda_stream_view stream,
                             memory_resource_args const& mr_args)
  : device_buffer{other.data(), other.size(), stream, mr_args}
{
}

device_buffer::device_buffer(device_buffer&& other) noexcept
  : _data{other._data},
    _size{other._size},
    _capacity{other._capacity},
    _stream{other.stream()},
    _mr{other._mr},
    _device{other._device},
    _host_mr{other._host_mr},
    _host_bounce_buffer{other._host_bounce_buffer}
{
  other._data     = nullptr;
  other._size     = 0;
  other._capacity = 0;
  other.set_stream(cuda_stream_view{});
  other._device             = cuda_device_id{-1};
  other._host_mr            = std::nullopt;
  other._host_bounce_buffer = std::nullopt;
}

device_buffer& device_buffer::operator=(device_buffer&& other) noexcept
{
  if (&other != this) {
    cuda_set_device_raii dev{_device};
    deallocate_async();

    _data     = other._data;
    _size     = other._size;
    _capacity = other._capacity;
    set_stream(other.stream());
    _mr                 = other._mr;
    _device             = other._device;
    _host_mr            = other._host_mr;
    _host_bounce_buffer = other._host_bounce_buffer;

    other._data     = nullptr;
    other._size     = 0;
    other._capacity = 0;
    other.set_stream(cuda_stream_view{});
    other._device             = cuda_device_id{-1};
    other._host_mr            = std::nullopt;
    other._host_bounce_buffer = std::nullopt;
  }
  return *this;
}

device_buffer::~device_buffer() noexcept
{
  cuda_set_device_raii dev{_device};
  deallocate_async();
  _stream = cuda_stream_view{};
}

void device_buffer::allocate_async(std::size_t bytes)
{
  auto const old_capacity = _capacity;
  _size                   = bytes;
  _capacity               = bytes;
  _data                   = (bytes > 0) ? _mr.allocate_async(bytes, stream()) : nullptr;

  // Resize host bounce buffer if needed
  if (_host_mr.has_value() && bytes > 0) {
    if (_host_bounce_buffer.has_value()) {
      _host_mr->deallocate(_host_bounce_buffer.value(), old_capacity);
    }
    _host_bounce_buffer = _host_mr->allocate(bytes);
  }
}

void device_buffer::deallocate_async() noexcept
{
  if (capacity() > 0) { _mr.deallocate_async(data(), capacity(), stream()); }

  // Deallocate host bounce buffer if it exists
  if (_host_bounce_buffer.has_value() && _host_mr.has_value()) {
    _host_mr->deallocate(_host_bounce_buffer.value(), capacity());
    _host_bounce_buffer = std::nullopt;
  }

  _size     = 0;
  _capacity = 0;
  _data     = nullptr;
}

void device_buffer::copy_async(void const* source, std::size_t bytes)
{
  if (bytes > 0) {
    RMM_EXPECTS(nullptr != source, "Invalid copy from nullptr.");
    RMM_EXPECTS(nullptr != _data, "Invalid copy to nullptr.");

    if (_host_bounce_buffer.has_value() && _host_mr.has_value()) {
      // If source is host memory, use bounce buffer for optimized transfer
      cudaPointerAttributes attributes;
      cudaError_t result = cudaPointerGetAttributes(&attributes, source);
      if (result == cudaSuccess && attributes.type == cudaMemoryTypeHost) {
        RMM_CUDA_TRY(cudaMemcpyAsync(
          _host_bounce_buffer.value(), source, bytes, cudaMemcpyHostToHost, stream().value()));
        RMM_CUDA_TRY(cudaMemcpyAsync(
          _data, _host_bounce_buffer.value(), bytes, cudaMemcpyHostToDevice, stream().value()));
        return;
      }
    }

    // Fallback to direct copy (device-to-device or host-to-device without bounce buffer)
    RMM_CUDA_TRY(cudaMemcpyAsync(_data, source, bytes, cudaMemcpyDefault, stream().value()));
  }
}

void device_buffer::reserve(std::size_t new_capacity, cuda_stream_view stream)
{
  set_stream(stream);
  if (new_capacity > capacity()) {
    cuda_set_device_raii dev{_device};

    device_buffer tmp;
    if (_host_mr.has_value()) {
      memory_resource_args args{_mr, _host_mr};
      tmp = device_buffer{new_capacity, stream, args};
    } else {
      tmp = device_buffer{new_capacity, stream, _mr};
    }

    auto const old_size = size();
    RMM_CUDA_TRY(cudaMemcpyAsync(
      tmp.data(), data(), size(), cudaMemcpyDefault, stream.value()));  // device-to-device copy
    *this = std::move(tmp);
    _size = old_size;
  }
}

void device_buffer::resize(std::size_t new_size, cuda_stream_view stream)
{
  set_stream(stream);
  // If the requested size is smaller than the current capacity, just update
  // the size without any allocations
  if (new_size <= capacity()) {
    _size = new_size;
  } else {
    cuda_set_device_raii dev{_device};

    device_buffer tmp;
    if (_host_mr.has_value()) {
      memory_resource_args args{_mr, _host_mr};
      tmp = device_buffer{new_size, stream, args};
    } else {
      tmp = device_buffer{new_size, stream, _mr};
    }

    RMM_CUDA_TRY(cudaMemcpyAsync(
      tmp.data(), data(), size(), cudaMemcpyDefault, stream.value()));  // device-to-device copy
    *this = std::move(tmp);
  }
}

void device_buffer::shrink_to_fit(cuda_stream_view stream)
{
  set_stream(stream);
  if (size() != capacity()) {
    cuda_set_device_raii dev{_device};
    // Invoke copy ctor on self which only copies `[0, size())` and swap it
    // with self. The temporary `device_buffer` will hold the old contents
    // which will then be destroyed
    device_buffer tmp;
    if (_host_mr.has_value()) {
      memory_resource_args args{_mr, _host_mr};
      tmp = device_buffer{*this, stream, args};
    } else {
      tmp = device_buffer{*this, stream, _mr};
    }
    *this = std::move(tmp);
  }
}

}  // namespace rmm
