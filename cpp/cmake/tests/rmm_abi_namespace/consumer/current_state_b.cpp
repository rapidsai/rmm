/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/per_device_resource.hpp>

extern "C" void const* current_state_b_map() { return &rmm::mr::detail::get_ref_map(); }
