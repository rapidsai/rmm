/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mr_ref_test.hpp"

#include <condition_variable>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

namespace rmm::test {

// Helper functions for multi-threaded tests

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

inline void async_allocate_loop(rmm::device_async_resource_ref ref,
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
    void* ptr        = ref.allocate(stream, size);
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

inline void async_deallocate_loop(rmm::device_async_resource_ref ref,
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
    ref.deallocate(stream, alloc.ptr, alloc.size);
  }

  // Work around for threads going away before cudaEvent has finished async processing
  cudaEventSynchronize(event);
}

inline void test_async_allocate_free_different_threads(rmm::device_async_resource_ref ref,
                                                       rmm::cuda_stream_view streamA,
                                                       rmm::cuda_stream_view streamB)
{
  constexpr std::size_t num_allocations{100};

  std::mutex mtx;
  std::condition_variable allocations_ready;
  std::list<allocation> allocations;
  cudaEvent_t event{};

  RMM_CUDA_TRY(cudaEventCreate(&event));

  std::thread producer(async_allocate_loop,
                       ref,
                       num_allocations,
                       std::ref(allocations),
                       std::ref(mtx),
                       std::ref(allocations_ready),
                       std::ref(event),
                       streamA);

  std::thread consumer(async_deallocate_loop,
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

}  // namespace rmm::test
