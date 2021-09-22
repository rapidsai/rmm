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
  rmm::cuda_event<rmm::EVENT_DEFAULT> event_b = std::move(event_a);
  EXPECT_FALSE(event_a.is_valid());
  EXPECT_TRUE(event_b.is_valid());
}

TEST_F(CudaEventTest, EventOrder)
{
  rmm::cuda_stream_view stream_a(rmm::cuda_stream_default);
  rmm::cuda_stream stream_b(rmm::STREAM_NON_BLOCKING);
  rmm::cuda_stream stream_c(rmm::STREAM_NON_BLOCKING);

  rmm::cuda_event<rmm::EVENT_DEFAULT> result_copied_back;
  rmm::cuda_event<rmm::EVENT_DISABLE_TIMING | rmm::EVENT_INTERPROCESS> memory_written_owner;
  rmm::cuda_event_view memory_written(memory_written_owner);
  rmm::cuda_event buf_created_owner;
  cudaEvent_t buf_created(buf_created_owner.value());

  const unsigned char expected = 42;
  unsigned char answer         = 0;

  // prepare the device memory
  rmm::device_scalar<unsigned char> buf(stream_c);
  stream_c.record_no_throw(buf_created);

  // set the expected result in the device memory
  stream_b.wait(buf_created);
  RMM_CUDA_TRY(cudaMemsetAsync(buf.data(), expected, 1, stream_b.value()));
  stream_b.record(memory_written, rmm::EVENT_RECORD_DEFAULT);

  // copy the written result back
  stream_a.wait(memory_written, rmm::EVENT_WAIT_DEFAULT);
  RMM_CUDA_TRY(cudaMemcpyAsync(&answer, buf.data(), 1, cudaMemcpyDeviceToHost, stream_a));
  stream_a.record(result_copied_back);

  // this must not affect the result, because memset happens after the data has been copied back.
  stream_c.wait(result_copied_back);
  RMM_CUDA_TRY(cudaMemsetAsync(buf.data(), expected + 1, 1, stream_c.value()));
  stream_c.record(memory_written);

  memory_written.wait();
  result_copied_back.wait();
  // At this moment, buf should contain incorrect data, but answer must be correct.
  EXPECT_EQ(expected, answer);

  // copy the wrong result back
  RMM_CUDA_TRY(cudaMemcpyAsync(&answer, buf.data(), 1, cudaMemcpyDeviceToHost, stream_a));
  stream_a.synchronize();
  EXPECT_NE(expected, answer);

  EXPECT_GE(result_copied_back.elapsed_time_since(buf_created), 0.0f);
}
