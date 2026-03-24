/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/detail/error.hpp>
#include <rmm/mr/sam_headroom_memory_resource.hpp>

namespace RMM_NAMESPACE {
namespace mr {

sam_headroom_memory_resource::sam_headroom_memory_resource(std::size_t headroom)
  : shared_base(cuda::mr::make_shared_resource<detail::sam_headroom_memory_resource_impl>(headroom))
{
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
