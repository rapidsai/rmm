/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <gtest/gtest-death-test.h>
#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <utility>

struct CudaStreamTest : public ::testing::Test {};

TEST_F(CudaStreamTest, Equality)
{
  rmm::cuda_stream stream_a;
  auto const view_a       = stream_a.view();
  auto const view_default = rmm::cuda_stream_view{};

  EXPECT_EQ(stream_a, view_a);
  EXPECT_NE(stream_a, view_default);
  EXPECT_EQ(view_default, rmm::cuda_stream_view{});
  EXPECT_EQ(view_default, rmm::cuda_stream_default);
  EXPECT_NE(view_a, rmm::cuda_stream());
  EXPECT_NE(stream_a, rmm::cuda_stream());

  rmm::device_buffer buff{};
  EXPECT_EQ(buff.stream(), view_default);

  EXPECT_NE(static_cast<cudaStream_t>(stream_a), rmm::cuda_stream_default.value());
}

TEST_F(CudaStreamTest, ImplicitConversionToStreamRef)
{
  rmm::cuda_stream stream;
  cuda::stream_ref ref = stream;
  EXPECT_EQ(ref.get(), stream.value());
}

TEST_F(CudaStreamTest, StreamRefConsistentWithView)
{
  rmm::cuda_stream stream;
  cuda::stream_ref ref_from_stream = stream;
  cuda::stream_ref ref_from_view   = stream.view();
  EXPECT_EQ(ref_from_stream, ref_from_view);
}

TEST_F(CudaStreamTest, MoveConstructor)
{
  rmm::cuda_stream stream_a;
  auto const view_a         = stream_a.view();
  rmm::cuda_stream stream_b = std::move(stream_a);
  // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move)
  EXPECT_FALSE(stream_a.is_valid());  // Any other operations on stream_a are UB, may segfault
  EXPECT_EQ(stream_b, view_a);
}

TEST_F(CudaStreamTest, TestStreamViewOstream)
{
  rmm::cuda_stream stream_a;
  rmm::cuda_stream_view view(stream_a);

  std::ostringstream oss;

  oss << view;

  std::ostringstream oss_expected;

  oss_expected << stream_a.value();

  EXPECT_EQ(oss.str(), oss_expected.str());
}

// Without this we don't get test coverage of ~stream_view, presumably because it is elided
TEST_F(CudaStreamTest, TestStreamViewDestructor)
{
  auto view = std::make_shared<rmm::cuda_stream_view>(cuda::stream_ref{cudaStreamPerThread});
  view->synchronize();
}

TEST_F(CudaStreamTest, TestSyncNoThrow)
{
  rmm::cuda_stream stream_a;
  EXPECT_NO_THROW(stream_a.synchronize_no_throw());
}

TEST_F(CudaStreamTest, TestCreateDefault)
{
  rmm::cuda_stream stream(rmm::cuda_stream::flags::sync_default);
  unsigned int flags;
  RMM_CUDA_TRY(cudaStreamGetFlags(stream.value(), &flags));
  EXPECT_EQ(flags, cudaStreamDefault);
}

TEST_F(CudaStreamTest, TestCreateNonBlocking)
{
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  unsigned int flags;
  RMM_CUDA_TRY(cudaStreamGetFlags(stream.value(), &flags));
  EXPECT_EQ(flags, cudaStreamNonBlocking);
}

#ifndef NDEBUG
using CudaStreamDeathTest = CudaStreamTest;

TEST_F(CudaStreamDeathTest, TestSyncNoThrow)
{
  auto test = []() {
    rmm::cuda_stream stream_a;
    cudaStreamDestroy(static_cast<cudaStream_t>(stream_a));
    // should assert here or in `~cuda_stream()`
    stream_a.synchronize_no_throw();
  };
  EXPECT_DEATH(test(), "");
}
#endif
