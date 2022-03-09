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
  static constexpr auto dlclose_destructor = [](void* handle) { ::dlclose(handle); };
  inline static std::unique_ptr<void, decltype(dlclose_destructor)> cuda_runtime_lib{
    nullptr, dlclose_destructor};

  static bool open_cuda_runtime()
  {
    if (!cuda_runtime_lib) {
      ::dlerror();
      const int major               = CUDART_VERSION / 1000;
      const std::string libname_ver = "libcudart.so." + std::to_string(major) + ".0";
      const std::string libname     = "libcudart.so";

      auto ptr = ::dlopen(libname_ver.c_str(), RTLD_LAZY);
      if (!ptr) { ::dlopen(libname.c_str(), RTLD_LAZY); }
      if (!ptr) { return false; }

      cuda_runtime_lib.reset(ptr);
    }
    return true;
  }

  template <typename... Args>
  using function_return_type = std::add_pointer_t<cudaError_t(Args...)>;

  template <typename... Args>
  static function_return_type<Args...> function(const char* func_name)
  {
    if (!open_cuda_runtime()) { return nullptr; }
    auto* handle = ::dlsym(cuda_runtime_lib.get(), func_name);
    if (!handle) { return nullptr; }
    auto function_ptr = reinterpret_cast<function_return_type<Args...>>(handle);
    return function_ptr;
  }
};


/**
 * @brief `async_alloc` bind to the Stream Ordered Memory Allocator functions
 * at runtime.
 *
 * This allows us rmm users to compile/link against CUDA 11.2+ and run with
 * < CUDA 11.2 runtime as these functions are found at call time
 */
struct async_alloc {

  static bool is_supported()
  {
#if defined(RMM_STATIC_CUDART)
    static bool has_support = (CUDART_VERSION >= 11020);
#else
    static bool has_support = dynamic_load_runtime::function<void*>("cudaFreeAsync") != nullptr;
#endif
    return has_support;
  }

#if defined(RMM_STATIC_CUDART)
  #define RMM_SYNC_ALLOC_WRAPPER(name)\
  template <typename... Args> \
  static cudaError_t name(Args... args) { \
    return ::name(args...); \
  }
#else
  #define RMM_SYNC_ALLOC_WRAPPER(name)\
  template <typename... Args> \
  static cudaError_t name(Args... args) { \
    static const auto func = dynamic_load_runtime::function<Args...>(#name); \
    return func(args...); \
  }
#endif

  RMM_SYNC_ALLOC_WRAPPER(cudaMemPoolCreate);
  RMM_SYNC_ALLOC_WRAPPER(cudaMemPoolSetAttribute);
  RMM_SYNC_ALLOC_WRAPPER(cudaMemPoolDestroy);
  RMM_SYNC_ALLOC_WRAPPER(cudaMallocFromPoolAsync);
  RMM_SYNC_ALLOC_WRAPPER(cudaFreeAsync);
};
}  // namespace rmm::detail
