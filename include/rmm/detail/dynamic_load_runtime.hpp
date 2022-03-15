/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <memory>

namespace rmm::detail {

/**
 * @brief `dynamic_load_runtime` loads the cuda runtime library at runtime
 *
 * By loading the cudart library at runtime we can use functions that
 * are added in newer minor versions of the cuda runtime.
 */
struct dynamic_load_runtime {
  static void* get_cuda_runtime_handle()
  {
    auto close_cudart = [](void* handle) { ::dlclose(handle); };
    auto open_cudart  = []() {
      ::dlerror();
      const int major               = CUDART_VERSION / 1000;
      const std::string libname_ver = "libcudart.so." + std::to_string(major) + ".0";
      const std::string libname     = "libcudart.so";

      auto ptr = ::dlopen(libname_ver.c_str(), RTLD_LAZY);
      if (!ptr) { ptr = ::dlopen(libname.c_str(), RTLD_LAZY); }
      if (ptr) { return ptr; }

      RMM_FAIL("Unable to dlopen cudart");
    };
    static std::unique_ptr<void, decltype(close_cudart)> cudart_handle{open_cudart(), close_cudart};
    return cudart_handle.get();
  }

  template <typename... Args>
  using funcion_sig = std::add_pointer_t<cudaError_t(Args...)>;

  template <typename signature>
  static signature function(const char* func_name)
  {
    auto* runtime = get_cuda_runtime_handle();
    auto* handle  = ::dlsym(runtime, func_name);
    if (!handle) { return nullptr; }
    auto* function_ptr = reinterpret_cast<signature>(handle);
    return function_ptr;
  }
};

#if CUDART_VERSION >= 11020  // 11.2 introduced cudaMallocAsync
/**
 * @brief `async_alloc` bind to the Stream Ordered Memory Allocator functions
 * at runtime.
 *
 * This allows RMM users to compile/link against CUDA 11.2+ and run with
 * < CUDA 11.2 runtime as these functions are found at call time
 */
struct async_alloc {
  static bool is_supported()
  {
#if defined(RMM_STATIC_CUDART)
    static bool runtime_supports_pool = (CUDART_VERSION >= 11020);
#else
    static bool runtime_supports_pool =
      dynamic_load_runtime::function<dynamic_load_runtime::funcion_sig<void*, cudaStream_t>>(
        "cudaFreeAsync") != nullptr;
#endif

    static auto driver_supports_pool{[] {
      int cuda_pool_supported{};
      auto result = cudaDeviceGetAttribute(&cuda_pool_supported,
                                           cudaDevAttrMemoryPoolsSupported,
                                           rmm::detail::current_device().value());
      return result == cudaSuccess and cuda_pool_supported == 1;
    }()};
    return runtime_supports_pool and driver_supports_pool;
  }

#if defined(RMM_STATIC_CUDART)
#define RMM_SYNC_ALLOC_WRAPPER(name, signature)              \
  template <typename... Args>                                \
  static cudaError_t name(Args... args)                      \
  {                                                          \
    static const auto func = static_cast<signature>(::name); \
    return func(args...);                                    \
  }
#else
#define RMM_SYNC_ALLOC_WRAPPER(name, signature)                                \
  template <typename... Args>                                                  \
  static cudaError_t name(Args... args)                                        \
  {                                                                            \
    static const auto func = dynamic_load_runtime::function<signature>(#name); \
    return func(args...);                                                      \
  }
#endif

  template <typename... Args>
  using cudart_sig = dynamic_load_runtime::funcion_sig<Args...>;

  using cudaMemPoolCreate_sig = cudart_sig<cudaMemPool_t*, const cudaMemPoolProps*>;
  RMM_SYNC_ALLOC_WRAPPER(cudaMemPoolCreate, cudaMemPoolCreate_sig);

  using cudaMemPoolSetAttribute_sig = cudart_sig<cudaMemPool_t, cudaMemPoolAttr, void*>;
  RMM_SYNC_ALLOC_WRAPPER(cudaMemPoolSetAttribute, cudaMemPoolSetAttribute_sig);

  using cudaMemPoolDestroy_sig = cudart_sig<cudaMemPool_t>;
  RMM_SYNC_ALLOC_WRAPPER(cudaMemPoolDestroy, cudaMemPoolDestroy_sig);

  using cudaMallocFromPoolAsync_sig = cudart_sig<void**, size_t, cudaMemPool_t, cudaStream_t>;
  RMM_SYNC_ALLOC_WRAPPER(cudaMallocFromPoolAsync, cudaMallocFromPoolAsync_sig);

  using cudaFreeAsync_sig = cudart_sig<void*, cudaStream_t>;
  RMM_SYNC_ALLOC_WRAPPER(cudaFreeAsync, cudaFreeAsync_sig);
};
#endif
}  // namespace rmm::detail
