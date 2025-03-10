/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

/**
 * @file libcudacxx_flag_test.cpp
 * @brief Test that verifies the error message when `LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE`
 * is not defined.
 *
 * This test is expected to fail to compile with a clear error message.
 * To run this test, you need to compile it separately without defining
 * `LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE`.
 *
 * Example:
 * `g++ -std=c++17 -I../../include libcudacxx_flag_test.cpp`
 */

// Include a header that requires LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
#include <rmm/detail/cuda_memory_resource.hpp>

int main() { return 0; }
