/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/arena_memory_resource.hpp>

namespace RMM_NAMESPACE {
namespace mr {

arena_memory_resource::arena_memory_resource(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
  std::optional<std::size_t> arena_size,
  bool dump_log_on_failure)
  : shared_base(cuda::mr::make_shared_resource<detail::arena_memory_resource_impl>(
      std::move(upstream), arena_size, dump_log_on_failure))
{
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
