/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/detail/callback_memory_resource_impl.hpp>

#include <utility>

namespace RMM_NAMESPACE {
namespace mr {

callback_memory_resource::callback_memory_resource(allocate_callback_t allocate_callback,
                                                   deallocate_callback_t deallocate_callback,
                                                   void* allocate_callback_arg,
                                                   void* deallocate_callback_arg)
  : shared_base(cuda::mr::make_shared_resource<detail::callback_memory_resource_impl>(
      std::move(allocate_callback),
      std::move(deallocate_callback),
      allocate_callback_arg,
      deallocate_callback_arg))
{
}

}  // namespace mr
}  // namespace RMM_NAMESPACE
