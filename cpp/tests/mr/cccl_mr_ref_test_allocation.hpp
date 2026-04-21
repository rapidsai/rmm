/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mr_ref_test.hpp"

namespace rmm::test {

/**
 * @brief Typed-test fixture for CCCL-style memory resource allocation tests.
 *
 * The Fixture parameter must be a ::testing::Test subclass providing:
 *   rmm::device_async_resource_ref ref
 *   rmm::cuda_stream stream
 */
template <typename Fixture>
struct CcclMrRefAllocationTest : public Fixture {};

TYPED_TEST_SUITE_P(CcclMrRefAllocationTest);

TYPED_TEST_P(CcclMrRefAllocationTest, AllocateDefault) { test_various_allocations(this->ref); }

TYPED_TEST_P(CcclMrRefAllocationTest, AllocateDefaultStream)
{
  test_various_async_allocations(this->ref, cuda_stream_view{});
}

TYPED_TEST_P(CcclMrRefAllocationTest, AllocateOnStream)
{
  test_various_async_allocations(this->ref, this->stream);
}

TYPED_TEST_P(CcclMrRefAllocationTest, RandomAllocations) { test_random_allocations(this->ref); }

TYPED_TEST_P(CcclMrRefAllocationTest, RandomAllocationsDefaultStream)
{
  test_random_async_allocations(
    this->ref, default_num_allocations, default_max_size, cuda_stream_view{});
}

TYPED_TEST_P(CcclMrRefAllocationTest, RandomAllocationsStream)
{
  test_random_async_allocations(this->ref, default_num_allocations, default_max_size, this->stream);
}

TYPED_TEST_P(CcclMrRefAllocationTest, MixedRandomAllocationFree)
{
  test_mixed_random_allocation_free(this->ref, default_max_size);
}

TYPED_TEST_P(CcclMrRefAllocationTest, MixedRandomAllocationFreeDefaultStream)
{
  test_mixed_random_async_allocation_free(this->ref, default_max_size, cuda_stream_view{});
}

TYPED_TEST_P(CcclMrRefAllocationTest, MixedRandomAllocationFreeStream)
{
  test_mixed_random_async_allocation_free(this->ref, default_max_size, this->stream);
}

REGISTER_TYPED_TEST_SUITE_P(CcclMrRefAllocationTest,
                            AllocateDefault,
                            AllocateDefaultStream,
                            AllocateOnStream,
                            RandomAllocations,
                            RandomAllocationsDefaultStream,
                            RandomAllocationsStream,
                            MixedRandomAllocationFree,
                            MixedRandomAllocationFreeDefaultStream,
                            MixedRandomAllocationFreeStream);

}  // namespace rmm::test
