/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#if !defined(RMM_DEPRECATE_MR_DEVICE_HEADERS) || RMM_DEPRECATE_MR_DEVICE_HEADERS
#pragma message( \
  "rmm/mr/device/detail/fixed_size_free_list.hpp is deprecated in 25.12 and will be removed in 26.02. Use rmm/mr/detail/fixed_size_free_list.hpp instead.")
#endif
#include <rmm/mr/detail/fixed_size_free_list.hpp>
