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

#include <rmm/detail/cnmem.h>
#include "device_memory_resource.hpp"

#include <cuda_runtime_api.h>
#include <cassert>
#include <exception>
#include <iostream>
#include <mutex>
#include <set>

namespace rmm {
namespace mr {
/**---------------------------------------------------------------------------*
 * @brief Memory resource that allocates/deallocates using the cnmem pool sub-allocator 
 * the cnmem pool sub-allocator for allocation/deallocation.
 *---------------------------------------------------------------------------**/
class cnmem_memory_resource final : public device_memory_resource {
 public:
  /**---------------------------------------------------------------------------*
   * @brief Construct a cnmem memory resource and allocate the initial device
   * memory pool

   * TODO Add constructor arguments for other CNMEM options/flags
   *
   * @param initial_pool_size Size, in bytes, of the intial pool size. When
   * zero, an implementation defined pool size is used.
   *---------------------------------------------------------------------------**/
  explicit cnmem_memory_resource(std::size_t initial_pool_size = 0) {
    cnmemDevice_t dev;
    // TODO Update exception
    if (cudaSuccess != cudaGetDevice(&(dev.device))) {
      throw std::runtime_error{"Failed to get CUDA device"};
    }
    // Note: cnmem defaults to half GPU memory
    dev.size = initial_pool_size;
    dev.numStreams = 1;
    cudaStream_t streams[1];
    streams[0] = 0;
    dev.streams = streams;
    dev.streamSizes = 0;
    unsigned flags = 0;
    // TODO Update exception
    auto status = cnmemInit(1, &dev, flags);
    if (CNMEM_STATUS_SUCCESS != status) {
      std::string msg = cnmemGetErrorString(status);
      throw std::runtime_error{"Failed to intialize cnmem: " + msg};
    }
  }

  ~cnmem_memory_resource() {
    auto status = cnmemFinalize();
#ifndef NDEBUG
    if (status != CNMEM_STATUS_SUCCESS) {
      std::cerr << "cnmemFinalize failed.\n";
    }
#endif
  }

  bool supports_streams() const noexcept override { return true; }

 private:
  /**---------------------------------------------------------------------------*
   * @brief Allocates memory of size at least \p bytes using cnmem.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `std::runtime_error` if cnmem failed to register the stream
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @return void* Pointer to the newly allocated memory
   *---------------------------------------------------------------------------**/
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override {
    register_stream(stream);
    void* p{nullptr};
    auto status = cnmemMalloc(&p, bytes, stream);
    if (CNMEM_STATUS_SUCCESS != status) {
#ifndef NDEBUG
      std::cerr << "cnmemMalloc failed\n";
#endif
      throw std::bad_alloc{};
    }
    return p;
  }

  /**---------------------------------------------------------------------------*
   * @brief Deallocate memory pointed to by \p p.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   *---------------------------------------------------------------------------**/
  void do_deallocate(void* p, std::size_t, cudaStream_t stream) override {
    auto status = cnmemFree(p, stream);
    if (CNMEM_STATUS_SUCCESS != status) {
#ifndef NDEBUG
      std::cerr << "cnmemFree failed \n";
#endif
    }
  }

  void register_stream(cudaStream_t stream) {
    // Don't register null stream with CNMEM
    if (stream != 0) {
      // TODO Probably don't want to have to take a lock for every memory
      // allocation
      std::lock_guard<std::mutex> lock(streams_mutex);
      auto result = registered_streams.insert(stream);

      if (result.second == true) {
        auto status = cnmemRegisterStream(stream);
        if (CNMEM_STATUS_SUCCESS != status) {
          throw std::runtime_error{"Falied to register stream with cnmem"};
        }
      }
    }
  }

  std::set<cudaStream_t> registered_streams{};
  std::mutex streams_mutex{};
};

}  // namespace mr
}  // namespace rmm
