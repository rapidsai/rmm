/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <rmm/cuda_event.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

struct CudaEventTest : public ::testing::Test {
};

TEST_F(CudaEventTest, MoveConstructor)
{
  rmm::cuda_event event_a;
  rmm::cuda_event event_b = std::move(event_a);
  EXPECT_FALSE(event_a.is_valid());
  EXPECT_TRUE(event_b.is_valid());
}

TEST_F(CudaEventTest, EventOrder)
{
  rmm::cuda_stream_view stream_a(rmm::cuda_stream_default);
  rmm::cuda_stream stream_b_owner(cudaStreamNonBlocking);
  cudaStream_t stream_b = stream_b_owner.value();
  rmm::cuda_stream stream_c(cudaStreamNonBlocking);

  rmm::cuda_event result_copied_back;
  rmm::cuda_event memory_written;
  rmm::cuda_event buf_created;

  const unsigned char expected = 42;
  unsigned char answer         = 0;

  // prepare the device memory
  rmm::device_scalar<unsigned char> buf(stream_c);
  buf_created.record_no_throw(stream_c);

  // set the expected result in the device memory
  buf_created.wait(stream_b);
  RMM_CUDA_TRY(cudaMemsetAsync(buf.data(), expected, 1, stream_b));
  memory_written.record(stream_b, 0);

  // copy the written result back
  memory_written.wait(stream_a, 0);
  RMM_CUDA_TRY(cudaMemcpyAsync(&answer, buf.data(), 1, cudaMemcpyDeviceToHost, stream_a));
  result_copied_back.record(stream_a);

  // this must not affect the result, because memset happens after the data has been copied back.
  result_copied_back.wait(stream_c);
  RMM_CUDA_TRY(cudaMemsetAsync(buf.data(), expected + 1, 1, stream_c.value()));
  memory_written.record(stream_c, 0);

  memory_written.wait();
  result_copied_back.wait();
  // At this moment, buf should contain incorrect data, but answer must be correct.
  EXPECT_EQ(expected, answer);

  // copy the wrong result back
  RMM_CUDA_TRY(cudaMemcpyAsync(&answer, buf.data(), 1, cudaMemcpyDeviceToHost, stream_a));
  stream_a.synchronize();
  EXPECT_NE(expected, answer);
}
