/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
