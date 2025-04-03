/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
/**
 * @brief Macro for suppressing __host__ / __device__ function markup
 * checks that the NVCC compiler does.
 *
 * At times it is useful to place rmm host only types inside containers
 * that work on both host and device. Doing so will generate warnings
 * of using a host only type inside a host / device type.
 *
 * This macro can be used to silence said warnings
 *
 */

// #pragma nv_exec_check_disable is only recognized by NVCC so verify
// that we have both the NVCC compiler and we are compiling a CUDA
// source
#if defined(__CUDACC__) && defined(__NVCC__)
#define RMM_EXEC_CHECK_DISABLE _Pragma("nv_exec_check_disable")
#else
#define RMM_EXEC_CHECK_DISABLE
#endif
