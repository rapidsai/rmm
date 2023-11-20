/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mr_ref_test.hpp"

#include <gtest/gtest.h>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuda/memory_resource>

#include <thread>
#include <vector>

namespace rmm::test {
namespace {

struct mr_ref_test_mt : public mr_ref_test {};

INSTANTIATE_TEST_CASE_P(MultiThreadResourceTests,
                        mr_ref_test_mt,
                        ::testing::Values(mr_factory{"CUDA", &make_cuda},
#ifdef RMM_CUDA_MALLOC_ASYNC_SUPPORT
                                          mr_factory{"CUDA_Async", &make_cuda_async},
#endif
                                          mr_factory{"Managed", &make_managed},
                                          mr_factory{"Pool", &make_pool},
                                          mr_factory{"Arena", &make_arena},
                                          mr_factory{"Binning", &make_binning}),
                        [](auto const& info) { return info.param.name; });

template <typename Task, typename... Arguments>
void spawn_n(std::size_t num_threads, Task task, Arguments&&... args)
{
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (std::size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back(std::thread(task, std::forward<Arguments>(args)...));
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

template <typename Task, typename... Arguments>
void spawn(Task task, Arguments&&... args)
{
  spawn_n(4, task, std::forward<Arguments>(args)...);
}

TEST_P(mr_ref_test_mt, Allocate) { spawn(test_various_allocations, this->ref); }

TEST_P(mr_ref_test_mt, AllocateDefaultStream)
{
  spawn(test_various_async_allocations, this->ref, rmm::cuda_stream_view{});
}

TEST_P(mr_ref_test_mt, AllocateOnStream)
{
  spawn(test_various_async_allocations, this->ref, this->stream.view());
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
        rmm::cuda_stream_view{});
}

TEST_P(mr_ref_test_mt, RandomAllocationsStream)
{
  spawn(test_random_async_allocations,
        this->ref,
        default_num_allocations,
        default_max_size,
        this->stream.view());
}

TEST_P(mr_ref_test_mt, MixedRandomAllocationFree)
{
  spawn(test_mixed_random_allocation_free, this->ref, default_max_size);
}

TEST_P(mr_ref_test_mt, MixedRandomAllocationFreeDefaultStream)
{
  spawn(
    test_mixed_random_async_allocation_free, this->ref, default_max_size, rmm::cuda_stream_view{});
}

TEST_P(mr_ref_test_mt, MixedRandomAllocationFreeStream)
{
  spawn(test_mixed_random_async_allocation_free, this->ref, default_max_size, this->stream.view());
}

void allocate_async_loop(async_resource_ref ref,
                         std::size_t num_allocations,
                         std::list<allocation>& allocations,
                         std::mutex& mtx,
                         std::condition_variable& allocations_ready,
                         cudaEvent_t& event,
                         rmm::cuda_stream_view stream)
{
  constexpr std::size_t max_size{1_MiB};

  std::default_random_engine generator;
  std::uniform_int_distribution<std::size_t> size_distribution(1, max_size);

  for (std::size_t i = 0; i < num_allocations; ++i) {
    std::size_t size = size_distribution(generator);
    void* ptr        = ref.allocate_async(size, stream);
    {
      std::lock_guard<std::mutex> lock(mtx);
      RMM_CUDA_TRY(cudaEventRecord(event, stream.value()));
      allocations.emplace_back(ptr, size);
    }
    allocations_ready.notify_one();
  }

  // Work around for threads going away before cudaEvent has finished async processing
  cudaEventSynchronize(event);
}

void deallocate_async_loop(async_resource_ref ref,
                           std::size_t num_allocations,
                           std::list<allocation>& allocations,
                           std::mutex& mtx,
                           std::condition_variable& allocations_ready,
                           cudaEvent_t& event,
                           rmm::cuda_stream_view stream)
{
  for (std::size_t i = 0; i < num_allocations; i++) {
    std::unique_lock lock(mtx);
    allocations_ready.wait(lock, [&allocations] { return !allocations.empty(); });
    RMM_CUDA_TRY(cudaStreamWaitEvent(stream.value(), event));
    allocation alloc = allocations.front();
    allocations.pop_front();
    ref.deallocate_async(alloc.ptr, alloc.size, stream);
  }

  // Work around for threads going away before cudaEvent has finished async processing
  cudaEventSynchronize(event);
}

void test_allocate_async_free_different_threads(async_resource_ref ref,
                                                rmm::cuda_stream_view streamA,
                                                rmm::cuda_stream_view streamB)
{
  constexpr std::size_t num_allocations{100};

  std::mutex mtx;
  std::condition_variable allocations_ready;
  std::list<allocation> allocations;
  cudaEvent_t event;

  RMM_CUDA_TRY(cudaEventCreate(&event));

  std::thread producer(allocate_async_loop,
                       ref,
                       num_allocations,
                       std::ref(allocations),
                       std::ref(mtx),
                       std::ref(allocations_ready),
                       std::ref(event),
                       streamA);

  std::thread consumer(deallocate_async_loop,
                       ref,
                       num_allocations,
                       std::ref(allocations),
                       std::ref(mtx),
                       std::ref(allocations_ready),
                       std::ref(event),
                       streamB);

  producer.join();
  consumer.join();

  RMM_CUDA_TRY(cudaEventDestroy(event));
}

TEST_P(mr_ref_test_mt, AllocFreeDifferentThreadsDefaultStream)
{
  test_allocate_async_free_different_threads(
    this->ref, rmm::cuda_stream_default, rmm::cuda_stream_default);
}

TEST_P(mr_ref_test_mt, AllocFreeDifferentThreadsPerThreadDefaultStream)
{
  test_allocate_async_free_different_threads(
    this->ref, rmm::cuda_stream_per_thread, rmm::cuda_stream_per_thread);
}

TEST_P(mr_ref_test_mt, AllocFreeDifferentThreadsSameStream)
{
  test_allocate_async_free_different_threads(this->ref, this->stream, this->stream);
}

TEST_P(mr_ref_test_mt, AllocFreeDifferentThreadsDifferentStream)
{
  rmm::cuda_stream streamB;
  test_allocate_async_free_different_threads(this->ref, this->stream, streamB);
  streamB.synchronize();
}

}  // namespace
}  // namespace rmm::test
