/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>

#include <cstddef>

std::size_t call_current_rmm_align_up(std::size_t value, std::size_t alignment)
{
  return rmm::align_up(value, alignment);
}
