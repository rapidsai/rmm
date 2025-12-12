/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mr_ref_test.hpp"

namespace rmm::test {

// Parameterized test definitions for mr_ref_allocation_test

TEST_P(mr_ref_allocation_test, AllocateDefault) { test_various_allocations(this->ref); }

TEST_P(mr_ref_allocation_test, AllocateDefaultStream)
{
  test_various_async_allocations(this->ref, cuda_stream_view{});
}

TEST_P(mr_ref_allocation_test, AllocateOnStream)
{
  test_various_async_allocations(this->ref, this->stream);
}

TEST_P(mr_ref_allocation_test, RandomAllocations) { test_random_allocations(this->ref); }

TEST_P(mr_ref_allocation_test, RandomAllocationsDefaultStream)
{
  test_random_async_allocations(
    this->ref, default_num_allocations, default_max_size, cuda_stream_view{});
}

TEST_P(mr_ref_allocation_test, RandomAllocationsStream)
{
  test_random_async_allocations(this->ref, default_num_allocations, default_max_size, this->stream);
}

TEST_P(mr_ref_allocation_test, MixedRandomAllocationFree)
{
  test_mixed_random_allocation_free(this->ref, default_max_size);
}

TEST_P(mr_ref_allocation_test, MixedRandomAllocationFreeDefaultStream)
{
  test_mixed_random_async_allocation_free(this->ref, default_max_size, cuda_stream_view{});
}

TEST_P(mr_ref_allocation_test, MixedRandomAllocationFreeStream)
{
  test_mixed_random_async_allocation_free(this->ref, default_max_size, this->stream);
}

}  // namespace rmm::test
