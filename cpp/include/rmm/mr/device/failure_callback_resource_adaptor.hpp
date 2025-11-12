/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#if !defined(RMM_DEPRECATE_MR_DEVICE_HEADERS) || RMM_DEPRECATE_MR_DEVICE_HEADERS
#pragma message( \
  "rmm/mr/device/failure_callback_resource_adaptor.hpp is deprecated in 25.12 and will be removed in 26.02. Use rmm/mr/failure_callback_resource_adaptor.hpp instead.")
#endif
#include <rmm/mr/failure_callback_resource_adaptor.hpp>
