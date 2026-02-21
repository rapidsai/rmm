/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mr_ref_test_mt_helpers.hpp"

namespace rmm::test {

/**
 * @brief Typed-test fixture for CCCL-style multi-threaded memory resource tests.
 *
 * The Fixture parameter must be a ::testing::Test subclass providing:
 *   rmm::device_async_resource_ref ref
 *   rmm::cuda_stream stream
 */
template <typename Fixture>
struct CcclMrRefTestMT : public Fixture {};

TYPED_TEST_SUITE_P(CcclMrRefTestMT);

TYPED_TEST_P(CcclMrRefTestMT, SetCurrentDeviceResourceRef_mt)
{
  rmm::mr::set_current_device_resource_ref(this->ref);
  test_get_current_device_resource_ref();

  int device;
  RMM_CUDA_TRY(cudaGetDevice(&device));

  spawn([device]() {
    RMM_CUDA_TRY(cudaSetDevice(device));
    test_get_current_device_resource_ref();
  });

  rmm::mr::reset_current_device_resource_ref();
  test_get_current_device_resource_ref();
}

TYPED_TEST_P(CcclMrRefTestMT, SetCurrentDeviceResourceRefPerThread_mt)
{
  int num_devices{};
  RMM_CUDA_TRY(cudaGetDeviceCount(&num_devices));

  std::vector<std::thread> threads;
  threads.reserve(num_devices);

  auto mr = this->ref;

  for (int i = 0; i < num_devices; ++i) {
    threads.emplace_back(
      [mr](auto dev_id) {
        RMM_CUDA_TRY(cudaSetDevice(dev_id));
        test_get_current_device_resource_ref();

        rmm::mr::set_current_device_resource_ref(mr);
        test_get_current_device_resource_ref();

        rmm::mr::reset_current_device_resource_ref();
        test_get_current_device_resource_ref();
      },
      i);
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

TYPED_TEST_P(CcclMrRefTestMT, Allocate)
{
  int device;
  RMM_CUDA_TRY(cudaGetDevice(&device));

  auto mr = this->ref;
  spawn([device, mr]() {
    RMM_CUDA_TRY(cudaSetDevice(device));
    test_various_allocations(mr);
  });
}

TYPED_TEST_P(CcclMrRefTestMT, AllocateDefaultStream)
{
  spawn(test_various_async_allocations, this->ref, rmm::cuda_stream_view{});
}

TYPED_TEST_P(CcclMrRefTestMT, AllocateOnStream)
{
  spawn(test_various_async_allocations, this->ref, this->stream.view());
}

TYPED_TEST_P(CcclMrRefTestMT, RandomAllocations)
{
  spawn(test_random_allocations, this->ref, default_num_allocations, default_max_size);
}

TYPED_TEST_P(CcclMrRefTestMT, RandomAllocationsDefaultStream)
{
  spawn(test_random_async_allocations,
        this->ref,
        default_num_allocations,
        default_max_size,
        rmm::cuda_stream_view{});
}

TYPED_TEST_P(CcclMrRefTestMT, RandomAllocationsStream)
{
  spawn(test_random_async_allocations,
        this->ref,
        default_num_allocations,
        default_max_size,
        this->stream.view());
}

TYPED_TEST_P(CcclMrRefTestMT, MixedRandomAllocationFree)
{
  spawn(test_mixed_random_allocation_free, this->ref, default_max_size);
}

TYPED_TEST_P(CcclMrRefTestMT, MixedRandomAllocationFreeDefaultStream)
{
  spawn(
    test_mixed_random_async_allocation_free, this->ref, default_max_size, rmm::cuda_stream_view{});
}

TYPED_TEST_P(CcclMrRefTestMT, MixedRandomAllocationFreeStream)
{
  spawn(test_mixed_random_async_allocation_free, this->ref, default_max_size, this->stream.view());
}

TYPED_TEST_P(CcclMrRefTestMT, AllocFreeDifferentThreadsDefaultStream)
{
  test_async_allocate_free_different_threads(
    this->ref, rmm::cuda_stream_default, rmm::cuda_stream_default);
}

TYPED_TEST_P(CcclMrRefTestMT, AllocFreeDifferentThreadsPerThreadDefaultStream)
{
  test_async_allocate_free_different_threads(
    this->ref, rmm::cuda_stream_per_thread, rmm::cuda_stream_per_thread);
}

TYPED_TEST_P(CcclMrRefTestMT, AllocFreeDifferentThreadsSameStream)
{
  test_async_allocate_free_different_threads(this->ref, this->stream, this->stream);
}

TYPED_TEST_P(CcclMrRefTestMT, AllocFreeDifferentThreadsDifferentStream)
{
  rmm::cuda_stream stream_b;
  test_async_allocate_free_different_threads(this->ref, this->stream, stream_b);
  stream_b.synchronize();
}

REGISTER_TYPED_TEST_SUITE_P(CcclMrRefTestMT,
                            SetCurrentDeviceResourceRef_mt,
                            SetCurrentDeviceResourceRefPerThread_mt,
                            Allocate,
                            AllocateDefaultStream,
                            AllocateOnStream,
                            RandomAllocations,
                            RandomAllocationsDefaultStream,
                            RandomAllocationsStream,
                            MixedRandomAllocationFree,
                            MixedRandomAllocationFreeDefaultStream,
                            MixedRandomAllocationFreeStream,
                            AllocFreeDifferentThreadsDefaultStream,
                            AllocFreeDifferentThreadsPerThreadDefaultStream,
                            AllocFreeDifferentThreadsSameStream,
                            AllocFreeDifferentThreadsDifferentStream);

}  // namespace rmm::test
