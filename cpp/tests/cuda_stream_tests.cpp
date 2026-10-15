/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/cuda_stream.hpp>
#include <rmm/detail/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <gtest/gtest-death-test.h>
#include <gtest/gtest.h>

#include <utility>

struct CudaStreamTest : public ::testing::Test {};

TEST_F(CudaStreamTest, Equality)
{
  rmm::cuda_stream stream_a;
  cuda::stream_ref const view_a = stream_a;
  auto const view_default       = cuda::stream_ref{cudaStream_t{cudaStreamDefault}};

  EXPECT_EQ(stream_a, view_a);
  EXPECT_NE(stream_a, view_default);
  EXPECT_NE(view_a, rmm::cuda_stream());

  rmm::device_buffer buff{};
  EXPECT_EQ(buff.stream(), view_default);

  EXPECT_NE(static_cast<cudaStream_t>(stream_a),
            cuda::stream_ref{cudaStream_t{cudaStreamDefault}}.get());
}

TEST_F(CudaStreamTest, ImplicitConversionToStreamRef)
{
  rmm::cuda_stream stream;
  cuda::stream_ref ref = stream;
  EXPECT_EQ(ref.get(), stream.value());
}

TEST_F(CudaStreamTest, IsDefaultStream)
{
  rmm::cuda_stream stream;

  EXPECT_FALSE(rmm::detail::is_default_stream(stream));
  EXPECT_TRUE(rmm::detail::is_default_stream(cuda::stream_ref{cudaStream_t{cudaStreamDefault}}));
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  EXPECT_FALSE(rmm::detail::is_default_stream(cuda::stream_ref{cudaStreamLegacy}));
  EXPECT_TRUE(rmm::detail::is_default_stream(cuda::stream_ref{cudaStreamPerThread}));
#else
  EXPECT_TRUE(rmm::detail::is_default_stream(cuda::stream_ref{cudaStreamLegacy}));
  EXPECT_FALSE(rmm::detail::is_default_stream(cuda::stream_ref{cudaStreamPerThread}));
#endif
}

TEST_F(CudaStreamTest, MoveConstructor)
{
  rmm::cuda_stream stream_a;
  cuda::stream_ref const view_a = stream_a;
  rmm::cuda_stream stream_b     = std::move(stream_a);
  // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move)
  EXPECT_FALSE(stream_a.is_valid());  // Any other operations on stream_a are UB, may segfault
  EXPECT_EQ(stream_b, view_a);
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
