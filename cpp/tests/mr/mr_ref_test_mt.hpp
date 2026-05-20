/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mr_ref_test_mt_helpers.hpp"

namespace rmm::test {

// Multi-threaded test fixture
struct mr_ref_test_mt : public mr_ref_test {};

// Parameterized test definitions for mr_ref_test_mt

TEST_P(mr_ref_test_mt, SetCurrentDeviceResourceRef_mt)
{
  // Single thread changes default resource, then multiple threads use it
  rmm::mr::set_current_device_resource(this->ref);
  test_get_current_device_resource_ref();

  int device;
  RMM_CUDA_TRY(cudaGetDevice(&device));

  spawn([device]() {
    RMM_CUDA_TRY(cudaSetDevice(device));
    test_get_current_device_resource_ref();
  });

  // Resetting default resource should reset to initial
  rmm::mr::reset_current_device_resource();
  // Verify reset worked by testing allocation with initial resource
  test_get_current_device_resource_ref();
}

TEST_P(mr_ref_test_mt, SetCurrentDeviceResourceRefPerThread_mt)
{
  int num_devices{};
  RMM_CUDA_TRY(cudaGetDeviceCount(&num_devices));

  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(num_devices));

  auto mr = this->ref;

  for (int i = 0; i < num_devices; ++i) {
    threads.emplace_back(
      [mr](auto dev_id) {
        RMM_CUDA_TRY(cudaSetDevice(dev_id));
        // Verify initial resource is functional
        test_get_current_device_resource_ref();

        rmm::mr::set_current_device_resource(mr);
        // Verify newly set resource is functional
        test_get_current_device_resource_ref();

        // Resetting current dev resource ref should restore initial resource
        rmm::mr::reset_current_device_resource();
        // Verify reset resource is functional
        test_get_current_device_resource_ref();
      },
      i);
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_P(mr_ref_test_mt, Allocate)
{
  int device;
  RMM_CUDA_TRY(cudaGetDevice(&device));

  auto mr = this->ref;
  spawn([device, mr]() {
    RMM_CUDA_TRY(cudaSetDevice(device));
    test_various_allocations(mr);
  });
}

TEST_P(mr_ref_test_mt, AllocateDefaultStream)
{
  spawn(test_various_async_allocations, this->ref, cuda::stream_ref{cudaStream_t{nullptr}});
}

TEST_P(mr_ref_test_mt, AllocateOnStream)
{
  spawn(test_various_async_allocations, this->ref, cuda::stream_ref{this->stream});
}

TEST_P(mr_ref_test_mt, RandomAllocations)
{
  spawn(test_random_allocations, this->ref, default_num_allocations, default_max_size);
}

TEST_P(mr_ref_test_mt, RandomAllocationsDefaultStream)
{
  spawn(test_random_async_allocations,
        this->ref,
        default_num_allocations,
        default_max_size,
        cuda::stream_ref{cudaStream_t{nullptr}});
}

TEST_P(mr_ref_test_mt, RandomAllocationsStream)
{
  spawn(test_random_async_allocations,
        this->ref,
        default_num_allocations,
        default_max_size,
        cuda::stream_ref{this->stream});
}

TEST_P(mr_ref_test_mt, MixedRandomAllocationFree)
{
  spawn(test_mixed_random_allocation_free, this->ref, default_max_size);
}

TEST_P(mr_ref_test_mt, MixedRandomAllocationFreeDefaultStream)
{
  spawn(test_mixed_random_async_allocation_free,
        this->ref,
        default_max_size,
        cuda::stream_ref{cudaStream_t{nullptr}});
}

TEST_P(mr_ref_test_mt, MixedRandomAllocationFreeStream)
{
  spawn(test_mixed_random_async_allocation_free,
        this->ref,
        default_max_size,
        cuda::stream_ref{this->stream});
}

TEST_P(mr_ref_test_mt, AllocFreeDifferentThreadsDefaultStream)
{
  test_async_allocate_free_different_threads(
    this->ref, cuda::stream_ref{cudaStream_t{nullptr}}, cuda::stream_ref{cudaStream_t{nullptr}});
}

TEST_P(mr_ref_test_mt, AllocFreeDifferentThreadsPerThreadDefaultStream)
{
  test_async_allocate_free_different_threads(
    this->ref, cuda::stream_ref{cudaStreamPerThread}, cuda::stream_ref{cudaStreamPerThread});
}

TEST_P(mr_ref_test_mt, AllocFreeDifferentThreadsSameStream)
{
  test_async_allocate_free_different_threads(
    this->ref, cuda::stream_ref{this->stream}, cuda::stream_ref{this->stream});
}

TEST_P(mr_ref_test_mt, AllocFreeDifferentThreadsDifferentStream)
{
  rmm::cuda_stream streamB;
  test_async_allocate_free_different_threads(
    this->ref, cuda::stream_ref{this->stream}, cuda::stream_ref{streamB});
  streamB.synchronize();
}

}  // namespace rmm::test
