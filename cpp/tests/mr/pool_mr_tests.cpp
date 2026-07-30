/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../byte_literals.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <thread>
#include <vector>

namespace rmm::test {
namespace {
using cuda_mr     = rmm::mr::cuda_memory_resource;
using pool_mr     = rmm::mr::pool_memory_resource;
using limiting_mr = rmm::mr::limiting_resource_adaptor;

TEST(PoolTest, ThrowMaxLessThanInitial)
{
  // Make sure first argument is enough larger than the second that alignment rounding doesn't
  // make them equal
  auto max_less_than_initial = []() {
    const auto initial{1024};
    const auto maximum{256};
    pool_mr mr{rmm::mr::get_current_device_resource_ref(), initial, maximum};
  };
  EXPECT_THROW(max_less_than_initial(), rmm::logic_error);
}

TEST(PoolTest, AllocateNinetyPercent)
{
  auto allocate_ninety = []() {
    auto const [free, total] = rmm::available_device_memory();
    (void)total;
    auto const ninety_percent_pool = rmm::percent_of_free_device_memory(90);
    pool_mr mr{rmm::mr::get_current_device_resource_ref(), ninety_percent_pool};
  };
  EXPECT_NO_THROW(allocate_ninety());
}

TEST(PoolTest, TwoLargeBuffers)
{
  auto two_large = []() {
    [[maybe_unused]] auto const [free, total] = rmm::available_device_memory();
    pool_mr mr{rmm::mr::get_current_device_resource_ref(), rmm::percent_of_free_device_memory(50)};
    auto* ptr1 = mr.allocate_sync(free / 4);
    auto* ptr2 = mr.allocate_sync(free / 4);
    mr.deallocate_sync(ptr1, free / 4);
    mr.deallocate_sync(ptr2, free / 4);
  };
  EXPECT_NO_THROW(two_large());
}

TEST(PoolTest, ForceGrowth)
{
  cuda_mr cuda;
  {
    auto const max_size{6000};
    limiting_mr limiter{cuda, max_size};
    pool_mr mr{limiter, 0};
    EXPECT_NO_THROW((void)mr.allocate_sync(1000));
    EXPECT_NO_THROW((void)mr.allocate_sync(4000));
    EXPECT_NO_THROW((void)mr.allocate_sync(500));
    EXPECT_THROW((void)mr.allocate_sync(2000), rmm::out_of_memory);  // too much
  }
  {
    // with max pool size
    auto const max_size{6000};
    limiting_mr limiter{cuda, max_size};
    pool_mr mr{limiter, 0, 8192};
    EXPECT_NO_THROW((void)mr.allocate_sync(1000));
    EXPECT_THROW((void)mr.allocate_sync(4000), rmm::out_of_memory);  // too much
    EXPECT_NO_THROW((void)mr.allocate_sync(500));
    EXPECT_NO_THROW((void)mr.allocate_sync(2000));  // fits
  }
}

TEST(PoolTest, DeletedStream)
{
  pool_mr mr{rmm::mr::get_current_device_resource_ref(), 0};
  cudaStream_t stream{};  // we don't use rmm::cuda_stream here to make destruction more explicit
  const int size = 10000;
  EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
  EXPECT_NO_THROW(rmm::device_buffer buff(size, cuda_stream_view{stream}, mr));
  EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  EXPECT_NO_THROW((void)mr.allocate_sync(size));
}

// Issue #527
TEST(PoolTest, InitialAndMaxPoolSizeEqual)
{
  EXPECT_NO_THROW([]() {
    pool_mr mr(rmm::mr::get_current_device_resource_ref(), 1000192, 1000192);
    (void)mr.allocate_sync(1000);
  }());
}

TEST(PoolTest, NonAlignedPoolSize)
{
  EXPECT_THROW(
    []() {
      pool_mr mr(rmm::mr::get_current_device_resource_ref(), 1000031, 1000192);
      (void)mr.allocate_sync(1000);
    }(),
    rmm::logic_error);

  EXPECT_THROW(
    []() {
      pool_mr mr(rmm::mr::get_current_device_resource_ref(), 1000192, 1000200);
      (void)mr.allocate_sync(1000);
    }(),
    rmm::logic_error);
}

TEST(PoolTest, UpstreamDoesntSupportMemInfo)
{
  pool_mr mr1{cuda_mr{}, 0};
  pool_mr mr2{mr1, 0};
  auto* ptr = mr2.allocate_sync(1024);
  mr2.deallocate_sync(ptr, 1024);
}

TEST(PoolTest, MultidevicePool)
{
  // Get the number of CUDA devices
  int num_devices = rmm::get_num_cuda_devices();

  // only run on multidevice systems
  if (num_devices >= 2) {
    rmm::mr::cuda_memory_resource general_mr;

    // initializing pool_memory_resource of multiple devices
    int devices      = 2;
    size_t pool_size = 1024;
    std::vector<pool_mr> mrs;

    for (int i = 0; i < devices; ++i) {
      RMM_CUDA_TRY(cudaSetDevice(i));
      auto mr = pool_mr{general_mr, pool_size, pool_size};
      rmm::mr::set_per_device_resource(rmm::cuda_device_id{i}, mr);
      mrs.emplace_back(mr);
    }

    {
      RMM_CUDA_TRY(cudaSetDevice(0));
      rmm::device_buffer buf_a(16, rmm::cuda_stream_per_thread, mrs[0]);

      {
        RMM_CUDA_TRY(cudaSetDevice(1));
        rmm::device_buffer buf_b(16, rmm::cuda_stream_per_thread, mrs[1]);
      }

      RMM_CUDA_TRY(cudaSetDevice(0));
    }
  }
}

// Host function used to stall a stream until the test releases it.
void CUDART_CB spin_until_released(void* flag)
{
  auto* released = static_cast<std::atomic<bool>*>(flag);
  while (!released->load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
}

// Invariant under test: a freed block may be handed to another stream only after that stream
// has been made dependent on all work that was in flight on the block when it was freed. This
// must hold transitively when the pool moves whole free lists between streams: if stream A's
// free list is merged into stream B's, and stream C later takes a block from B's list, C must
// still be ordered after A's outstanding work on that block, even though C only synchronizes
// with B's bookkeeping event.
//
// The test builds the shortest chain that exercises this, using three streams:
//   stream A frees block X while a write of `pattern_a` to X is still pending on A (A is
//   stalled by a host function, so the write cannot complete until the test releases it);
//   stream B requests an allocation that the pool can only attempt by merging A's free list
//   (containing X) into B's; the request itself fails, leaving X in B's list;
//   stream C then allocates X out of B's list and writes `pattern_c` to it.
// C's write must be stream-ordered after A's, so X must read back as `pattern_c` once both
// streams have drained, regardless of timing. A probe event additionally checks that C's
// write cannot complete while A is still stalled.
TEST(PoolTest, CrossStreamStealAfterMergeWaitsForDonorStream)
{
  constexpr std::size_t block_size{1_MiB};
  constexpr std::size_t pool_size{8_MiB};
  constexpr int pattern_a{0xAA};
  constexpr int pattern_c{0xBB};

  pool_mr mr{rmm::mr::get_current_device_resource_ref(), pool_size, pool_size};
  rmm::device_async_resource_ref ref{mr};

  rmm::cuda_stream stream_a;
  rmm::cuda_stream stream_b;
  rmm::cuda_stream stream_c;

  // X is carved from the front of the pool. The separator, allocated right behind it and held
  // for the whole test, prevents X from coalescing with the rest of the pool's free memory.
  void* ptr_x     = ref.allocate(stream_a.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  void* separator = ref.allocate(stream_b.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  // Stall stream A, enqueue a write to X behind the stall, then free X on A. The pool records
  // A's event behind the pending write, so any consumer that waits on A's event cannot touch X
  // before the write completes.
  std::atomic<bool> release{false};

  // Unblock the callback during unwinding so resource teardown cannot wait on it indefinitely.
  struct release_on_exit {
    std::atomic<bool>& flag;

    ~release_on_exit() { flag.store(true, std::memory_order_release); }
  };

  release_on_exit unblock{release};
  RMM_CUDA_TRY(cudaLaunchHostFunc(stream_a.value(), spin_until_released, &release));
  RMM_CUDA_TRY(cudaMemsetAsync(ptr_x, pattern_a, block_size, stream_a.value()));
  ref.deallocate(stream_a.view(), ptr_x, block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  // 6.5 MiB exceeds every individual free block (X: 1 MiB, rest of the pool: 6 MiB) and, since
  // the separator prevents coalescing, the merged list too; the pool is at its maximum size, so
  // this throws -- but only after merging A's free list (with X in it) into B's.
  constexpr auto unsatisfiable = 6_MiB + block_size / 2;
  EXPECT_THROW((void)ref.allocate(stream_b.view(), unsatisfiable, rmm::CUDA_ALLOCATION_ALIGNMENT),
               rmm::out_of_memory);

  // Steal X from B's free list on a third stream and overwrite it. This waits only on B's
  // event, which must have been recorded behind the wait on A's event during the merge above.
  void* ptr_c = ref.allocate(stream_c.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_EQ(ptr_c, ptr_x);  // best fit: the stolen block is exactly X
  RMM_CUDA_TRY(cudaMemsetAsync(ptr_c, pattern_c, block_size, stream_c.value()));

  // Probe whether C's write can complete while A is still stalled. With the dependency chain
  // intact it cannot -- C's stream is ordered behind A's pending work -- so the probe must
  // still be pending when the deadline expires; the deadline only bounds the poll and does not
  // affect correctness. If the chain is broken, the write completes almost immediately and the
  // poll observes it. (An unbounded poll would hang here on a correct implementation.)
  cudaEvent_t probe{};
  RMM_CUDA_TRY(cudaEventCreateWithFlags(&probe, cudaEventDisableTiming));
  RMM_CUDA_TRY(cudaEventRecord(probe, stream_c.value()));
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  auto probe_status   = cudaEventQuery(probe);
  while (probe_status == cudaErrorNotReady && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
    probe_status = cudaEventQuery(probe);
  }
  EXPECT_EQ(probe_status, cudaErrorNotReady);

  // Release stream A and drain both streams; only now may C's write complete.
  release.store(true, std::memory_order_release);
  stream_a.synchronize();
  stream_c.synchronize();
  RMM_CUDA_TRY(cudaEventDestroy(probe));

  // Stream C's write is ordered after stream A's, so it must win.
  std::vector<unsigned char> host(block_size);
  RMM_CUDA_TRY(cudaMemcpy(host.data(), ptr_c, block_size, cudaMemcpyDefault));
  EXPECT_TRUE(
    std::all_of(host.cbegin(), host.cend(), [](unsigned char byte) { return byte == pattern_c; }));

  ref.deallocate(stream_c.view(), ptr_c, block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  ref.deallocate(stream_b.view(), separator, block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

class PoolMemoryResourceTest : public ::testing::Test {
 protected:
  rmm::mr::pool_memory_resource pool{rmm::mr::get_current_device_resource_ref(), 1024 * 1024};
};

TEST_F(PoolMemoryResourceTest, GetUpstreamResource)
{
  [[maybe_unused]] auto ref = pool.get_upstream_resource();
}

TEST_F(PoolMemoryResourceTest, AllocateDeallocate)
{
  constexpr std::size_t size{4096};
  auto* ptr = pool.allocate_sync(size);
  EXPECT_NE(ptr, nullptr);
  EXPECT_NO_THROW(pool.deallocate_sync(ptr, size));
}

TEST_F(PoolMemoryResourceTest, SharedOwnership)
{
  auto copy = pool;  // copy shares the same underlying state
  constexpr std::size_t size{4096};
  auto* ptr = pool.allocate_sync(size);
  EXPECT_NE(ptr, nullptr);
  EXPECT_NO_THROW(copy.deallocate_sync(ptr, size));  // deallocate through the copy
}

TEST_F(PoolMemoryResourceTest, Equality)
{
  auto copy = pool;
  EXPECT_EQ(pool, copy);

  rmm::mr::pool_memory_resource other{rmm::mr::get_current_device_resource_ref(), 1024 * 1024};
  EXPECT_NE(pool, other);
}

TEST_F(PoolMemoryResourceTest, PoolSize) { EXPECT_GE(pool.pool_size(), 1024 * 1024); }

}  // namespace

namespace test_properties {

// static property checks
static_assert(cuda::mr::resource_with<rmm::mr::pool_memory_resource, cuda::mr::device_accessible>);

}  // namespace test_properties

}  // namespace rmm::test
