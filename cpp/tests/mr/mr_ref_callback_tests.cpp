/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cccl_mr_ref_test_allocation.hpp"
#include "cccl_mr_ref_test_basic.hpp"
#include "cccl_mr_ref_test_mt.hpp"

#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

namespace rmm::test {

struct CallbackMRFixture : public ::testing::Test {
  rmm::mr::cuda_memory_resource cuda{};
  rmm::device_async_resource_ref upstream{cuda};

  rmm::mr::callback_memory_resource mr{
    [](std::size_t bytes, rmm::cuda_stream_view stream, void* arg) {
      return static_cast<rmm::device_async_resource_ref*>(arg)->allocate(stream, bytes);
    },
    [](void* ptr, std::size_t bytes, rmm::cuda_stream_view stream, void* arg) {
      static_cast<rmm::device_async_resource_ref*>(arg)->deallocate(stream, ptr, bytes);
    },
    &upstream,
    &upstream};

  rmm::device_async_resource_ref ref{mr};
  rmm::cuda_stream stream{};
};

INSTANTIATE_TYPED_TEST_SUITE_P(CallbackMR, CcclMrRefTest, CallbackMRFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(CallbackMR, CcclMrRefAllocationTest, CallbackMRFixture);
INSTANTIATE_TYPED_TEST_SUITE_P(CallbackMR, CcclMrRefTestMT, CallbackMRFixture);

}  // namespace rmm::test
