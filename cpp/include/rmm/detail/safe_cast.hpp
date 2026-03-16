/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <limits>
#include <stdexcept>
#include <type_traits>

namespace rmm::detail {

/**
 * @brief Checked narrowing/sign-converting cast.
 *
 * Casts `value` to type `To`, throwing `std::overflow_error` if the value
 * is not representable in `To`.
 */
template <typename To, typename From>
[[nodiscard]] constexpr To safe_cast(From value)
{
  static_assert(std::is_integral_v<From> && std::is_integral_v<To>,
                "safe_cast is only defined for integral types");
  if constexpr (std::is_signed_v<From> && std::is_unsigned_v<To>) {
    if (value < From{0} || static_cast<std::make_unsigned_t<From>>(value) >
                             std::numeric_limits<To>::max()) {
      throw std::overflow_error("rmm::detail::safe_cast: value out of range for destination type");
    }
  } else if constexpr (std::is_unsigned_v<From> && std::is_signed_v<To>) {
    if (value > static_cast<From>(std::numeric_limits<To>::max())) {
      throw std::overflow_error("rmm::detail::safe_cast: value out of range for destination type");
    }
  } else if constexpr (sizeof(From) > sizeof(To)) {
    if (value < static_cast<From>(std::numeric_limits<To>::min()) ||
        value > static_cast<From>(std::numeric_limits<To>::max())) {
      throw std::overflow_error("rmm::detail::safe_cast: value out of range for destination type");
    }
  }
  return static_cast<To>(value);
}

}  // namespace rmm::detail
