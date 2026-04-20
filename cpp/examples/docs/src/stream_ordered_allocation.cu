/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Code examples for docs/user_guide/stream_ordered_allocation.md
//
// Include directives that appear inside function bodies are intentional:
// they are no-ops (headers use #pragma once) and exist so that
// literalinclude snippets display the includes alongside the code.

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

__global__ void trivial_kernel(void* data) {}

void how_it_works()
{
  dim3 grid(1), block(1);

  // clang-format off
  // [how-it-works]
  #include <rmm/mr/cuda_async_memory_resource.hpp>
  #include <rmm/device_buffer.hpp>

  rmm::mr::cuda_async_memory_resource mr;
  rmm::cuda_stream stream;
  rmm::device_buffer buffer(1000, stream.view(), mr);

  // buffer.data() is usable immediately in stream-ordered operations
  trivial_kernel<<<grid, block, 0, stream.value()>>>(buffer.data());
  // [/how-it-works]
  // clang-format on

  stream.synchronize();
}

void reading_results()
{
  // clang-format off
  // [reading-results]
  rmm::mr::cuda_async_memory_resource mr;
  rmm::cuda_stream stream;
  rmm::device_buffer d_buf(1000 * sizeof(float), stream.view(), mr);

  // Launch kernel that writes to d_buf on stream ...

  // Copy results to host on the same stream
  std::vector<float> h_buf(1000);
  cudaMemcpyAsync(h_buf.data(), d_buf.data(), d_buf.size(),
                  cudaMemcpyDeviceToHost, stream.value());

  // Synchronize before reading h_buf on the CPU
  stream.synchronize();
  // [/reading-results]
  // clang-format on
}

void cross_stream()
{
  dim3 grid(1), block(1);

  // clang-format off
  // [cross-stream]
  #include <rmm/mr/cuda_async_memory_resource.hpp>
  #include <rmm/device_buffer.hpp>

  rmm::mr::cuda_async_memory_resource mr;
  rmm::cuda_stream stream_a;
  rmm::cuda_stream stream_b;

  rmm::device_buffer buffer(1000, stream_a.view(), mr);

  // Record an event after the allocation on stream_a
  cudaEvent_t event;
  cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  cudaEventRecord(event, stream_a.value());

  // stream_b waits for the event — no CPU synchronization needed
  cudaStreamWaitEvent(stream_b.value(), event);

  // Now safe to use buffer.data() in operations on stream_b
  trivial_kernel<<<grid, block, 0, stream_b.value()>>>(buffer.data());

  cudaEventDestroy(event);
  // [/cross-stream]
  // clang-format on

  stream_b.synchronize();
}

void buffer_lifetime()
{
  dim3 grid(1), block(1);

  // clang-format off
  // [buffer-lifetime]
  rmm::mr::cuda_async_memory_resource mr;
  rmm::cuda_stream stream_a;
  rmm::cuda_stream stream_b;

  rmm::device_buffer buffer(1000, stream_a.view(), mr);

  // Make stream_b wait for the allocation on stream_a
  cudaEvent_t alloc_event;
  cudaEventCreateWithFlags(&alloc_event, cudaEventDisableTiming);
  cudaEventRecord(alloc_event, stream_a.value());
  cudaStreamWaitEvent(stream_b.value(), alloc_event);

  // Use buffer on stream_b
  trivial_kernel<<<grid, block, 0, stream_b.value()>>>(buffer.data());

  // Before destroying buffer, make stream_a wait for stream_b's work
  cudaEvent_t done_event;
  cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming);
  cudaEventRecord(done_event, stream_b.value());
  cudaStreamWaitEvent(stream_a.value(), done_event);

  // Now safe to destroy buffer — deallocation on stream_a is ordered after the kernel on stream_b
  buffer = rmm::device_buffer{};

  cudaEventDestroy(alloc_event);
  cudaEventDestroy(done_event);
  // [/buffer-lifetime]
  // clang-format on
}

int main()
{
  how_it_works();
  reading_results();
  cross_stream();
  buffer_lifetime();

  std::cout << "All stream_ordered_allocation examples passed.\n";
  return 0;
}
