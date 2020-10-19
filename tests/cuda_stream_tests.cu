/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "gtest/gtest.h"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

struct CudaStreamTest : public ::testing::Test {
  cudaStream_t stream{};

  void SetUp() override { EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream)); }

  void TearDown() override { EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream)); };
};

TEST_F(CudaStreamTest, SetDefault)
{
  auto initial = rmm::set_default_stream(rmm::cuda_stream_view{this->stream});

  auto new_default = rmm::get_default_stream();
  EXPECT_EQ(this->stream, new_default);
  EXPECT_NE(initial, new_default);

  rmm::device_buffer buff(0);
  EXPECT_EQ(buff.stream(), new_default);

  rmm::set_default_stream(initial);
  EXPECT_EQ(rmm::get_default_stream(), initial);
}
