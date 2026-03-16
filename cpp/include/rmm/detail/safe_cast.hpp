/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/detail/error.hpp>

#include <type_traits>

namespace rmm::detail {

/**
 * @brief Checked narrowing/sign-converting cast.
 *
 * Casts `value` to type `To`, asserting at runtime that the value is
 * representable in `To`. In release builds the assertion compiles away when
 * the compiler can prove it is always true.
 */
template <typename To, typename From>
[[nodiscard]] constexpr To safe_cast(From value)
{
  static_assert(std::is_integral_v<From> && std::is_integral_v<To>,
                "safe_cast is only defined for integral types");
  if constexpr (std::is_signed_v<From> && std::is_unsigned_v<To>) {
    RMM_EXPECTS(value >= From{0}, "safe_cast: negative value cannot be represented as unsigned");
  }
  return static_cast<To>(value);
}

}  // namespace rmm::detail
