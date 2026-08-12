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
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/detail/coalescing_free_list.hpp>
#include <rmm/mr/detail/indexed_coalescing_free_list.hpp>
#include <rmm/mr/indexed_pool_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace rmm::test {
namespace {
using cuda_mr     = rmm::mr::cuda_memory_resource;
using pool_mr     = rmm::mr::pool_memory_resource;
using limiting_mr = rmm::mr::limiting_resource_adaptor;

class host_func_gate {
 public:
  void wait()
  {
    std::unique_lock<std::mutex> lock{mutex_};
    EXPECT_TRUE(condition_.wait_for(lock, std::chrono::seconds{10}, [this] { return released_; }));
    complete_.store(true);
  }

  void release()
  {
    {
      std::lock_guard<std::mutex> lock{mutex_};
      released_ = true;
    }
    condition_.notify_one();
  }

  bool complete() const { return complete_.load(); }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  bool released_{false};
  std::atomic<bool> complete_{false};
};

class host_func_gate_release_guard {
 public:
  explicit host_func_gate_release_guard(host_func_gate& gate) : gate_{gate} {}

  ~host_func_gate_release_guard() { gate_.release(); }

 private:
  host_func_gate& gate_;
};

class delayed_async_memory_resource {
 public:
  explicit delayed_async_memory_resource(std::shared_ptr<std::atomic<bool>> release)
    : release_{std::move(release)}
  {
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    auto* ptr = upstream_.allocate(stream, bytes, alignment);
    RMM_CUDA_TRY(cudaLaunchHostFunc(
      stream.get(),
      [](void* flag) {
        auto* release = static_cast<std::atomic<bool>*>(flag);
        while (!release->load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
      },
      release_.get()));
    return ptr;
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  { upstream_.deallocate(stream, ptr, bytes, alignment); }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  { return upstream_.allocate_sync(bytes, alignment); }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  { upstream_.deallocate_sync(ptr, bytes, alignment); }

  bool operator==(delayed_async_memory_resource const& other) const noexcept
  { return release_ == other.release_; }

  bool operator!=(delayed_async_memory_resource const& other) const noexcept
  { return !(*this == other); }

  RMM_CONSTEXPR_FRIEND void get_property(delayed_async_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 private:
  cuda_mr upstream_;
  std::shared_ptr<std::atomic<bool>> release_;
};

static_assert(cuda::mr::resource_with<delayed_async_memory_resource, cuda::mr::device_accessible>);

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

// Issue #1957
TEST(PoolTest, AllocateMaxWithInitialPoolSize)
{
  EXPECT_NO_THROW([]() {
    pool_mr mr(rmm::mr::get_current_device_resource_ref(), 256, 1024);
    auto* ptr = mr.allocate_sync(1024);
    mr.deallocate_sync(ptr, 1024);
  }());
}

TEST(PoolTest, AllocateMaximumDirectlyWithZeroInitialPoolSize)
{
  EXPECT_NO_THROW([]() {
    pool_mr mr(rmm::mr::get_current_device_resource_ref(), 0, 1024);
    auto* ptr = mr.allocate_sync(1024);
    mr.deallocate_sync(ptr, 1024);
  }());
}

// Issue #1957: a partially-used upstream block must never be reclaimed.
TEST(PoolTest, PartialUseBlockNotReclaimed)
{
  pool_mr mr(rmm::mr::get_current_device_resource_ref(), 1024, 2048);
  // Splits the 1024B upstream block, leaving it partly in use.
  auto* ptr = mr.allocate_sync(512);
  // The 1024B block is not fully free, so it cannot be reclaimed; the pool still cannot grow to
  // satisfy 2048B under the 2048B cap.
  EXPECT_THROW((void)mr.allocate_sync(2048), rmm::out_of_memory);
  // The held pointer remained valid throughout.
  EXPECT_NO_THROW(mr.deallocate_sync(ptr, 512));
}

TEST(PoolTest, MissingReclaimCandidateDoesNotWaitForPendingWork)
{
  cuda_mr upstream;
  pool_mr mr{upstream, 1024, 1024};
  auto* held_ptr = mr.allocate_sync(512);

  host_func_gate prior_work;
  rmm::cuda_stream source;
  host_func_gate_release_guard const release_prior_work{prior_work};
  auto* source_ptr = mr.allocate(source.view(), 256, rmm::CUDA_ALLOCATION_ALIGNMENT);
  RMM_CUDA_TRY(cudaLaunchHostFunc(
    source.value(), [](void* data) { static_cast<host_func_gate*>(data)->wait(); }, &prior_work));
  mr.deallocate(source.view(), source_ptr, 256, rmm::CUDA_ALLOCATION_ALIGNMENT);

  rmm::cuda_stream destination;
  auto const failed_without_waiting = [&]() {
    try {
      auto* ptr = mr.allocate(destination.view(), 1024, rmm::CUDA_ALLOCATION_ALIGNMENT);
      mr.deallocate(destination.view(), ptr, 1024, rmm::CUDA_ALLOCATION_ALIGNMENT);
      return false;
    } catch (rmm::out_of_memory const&) {
      return !prior_work.complete();
    }
  }();

  prior_work.release();
  source.synchronize();
  destination.synchronize();
  mr.deallocate_sync(held_ptr, 512);

  EXPECT_TRUE(failed_without_waiting);
  EXPECT_TRUE(prior_work.complete());
}

// Issue #1957: a request larger than the maximum pool size throws (and reclaim terminates).
TEST(PoolTest, AllocateLargerThanMaxThrows)
{
  pool_mr mr(rmm::mr::get_current_device_resource_ref(), 1024, 1024);
  EXPECT_THROW((void)mr.allocate_sync(2048), rmm::out_of_memory);

  EXPECT_EQ(mr.pool_size(), 1024);
  EXPECT_NO_THROW([](pool_mr& resource) {
    auto* ptr = resource.allocate_sync(1024);
    resource.deallocate_sync(ptr, 1024);
  }(mr));
}

// Issue #1957: reclaim a block sitting in a non-default stream's free list.
TEST(PoolTest, ReclaimAcrossStreams)
{
  pool_mr mr(rmm::mr::get_current_device_resource_ref(), 256, 1024);
  rmm::cuda_stream stream;
  {
    // Allocate and free 256B on a non-default stream so the freed block lands in that stream's
    // free list.
    rmm::device_buffer buf(256, stream.view(), mr);
  }
  // Growing to 1024B requires reclaiming the entirely-free cross-stream block.
  EXPECT_NO_THROW([&]() {
    auto* ptr = mr.allocate_sync(1024);
    mr.deallocate_sync(ptr, 1024);
  }());
}

TEST(PoolTest, ReclaimIsStreamOrderedOnSameStream)
{
  rmm::mr::cuda_async_memory_resource upstream;
  pool_mr mr{upstream, 256, 1024};
  host_func_gate prior_work;
  host_func_gate_release_guard const release_prior_work{prior_work};
  rmm::cuda_stream stream;

  auto* source_ptr = mr.allocate(stream.view(), 256, rmm::CUDA_ALLOCATION_ALIGNMENT);
  RMM_CUDA_TRY(cudaLaunchHostFunc(
    stream.value(), [](void* data) { static_cast<host_func_gate*>(data)->wait(); }, &prior_work));
  mr.deallocate(stream.view(), source_ptr, 256, rmm::CUDA_ALLOCATION_ALIGNMENT);

  auto* destination_ptr = mr.allocate(stream.view(), 1024, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto const reclaim_returned_without_waiting = !prior_work.complete();

  prior_work.release();
  mr.deallocate(stream.view(), destination_ptr, 1024, rmm::CUDA_ALLOCATION_ALIGNMENT);
  stream.synchronize();

  EXPECT_TRUE(reclaim_returned_without_waiting);
  EXPECT_TRUE(prior_work.complete());
}

TEST(PoolTest, ReclaimIsStreamOrderedWithMergedPerThreadDefaultStream)
{
  rmm::mr::cuda_async_memory_resource upstream;
  pool_mr mr{upstream, 256, 1024};
  host_func_gate prior_work;
  host_func_gate_release_guard const release_prior_work{prior_work};

  auto* source_ptr = mr.allocate(rmm::cuda_stream_per_thread, 256, rmm::CUDA_ALLOCATION_ALIGNMENT);
  RMM_CUDA_TRY(cudaLaunchHostFunc(
    rmm::cuda_stream_per_thread.value(),
    [](void* data) { static_cast<host_func_gate*>(data)->wait(); },
    &prior_work));
  mr.deallocate(rmm::cuda_stream_per_thread, source_ptr, 256, rmm::CUDA_ALLOCATION_ALIGNMENT);

  rmm::cuda_stream destination;
  auto* destination_ptr = mr.allocate(destination.view(), 1024, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto const reclaim_returned_without_waiting = !prior_work.complete();

  prior_work.release();
  mr.deallocate(destination.view(), destination_ptr, 1024, rmm::CUDA_ALLOCATION_ALIGNMENT);
  destination.synchronize();

  EXPECT_TRUE(reclaim_returned_without_waiting);
  EXPECT_TRUE(prior_work.complete());
}

TEST(PoolTest, ReclaimsToMaximumWithCudaAsyncUpstreamAcrossStreams)
{
  rmm::mr::cuda_async_memory_resource upstream;
  pool_mr mr{upstream, 256, 1024};
  rmm::cuda_stream source;
  {
    rmm::device_buffer source_block{256, source.view(), mr};
  }

  auto* ptr                      = mr.allocate_sync(1024);
  auto const reclaimed_pool_size = mr.pool_size();
  mr.deallocate_sync(ptr, 1024);

  EXPECT_EQ(reclaimed_pool_size, 1024);
}

TEST(IndexedPoolTest, AllocateMaxWithInitialPoolSize)
{
  rmm::mr::indexed_pool_memory_resource mr{
    rmm::mr::get_current_device_resource_ref(), 256, 1024};
  auto* ptr = mr.allocate_sync(1024);
  EXPECT_NE(ptr, nullptr);
  EXPECT_NO_THROW(mr.deallocate_sync(ptr, 1024));
}

TEST(IndexedPoolTest, PartialUseBlockNotReclaimed)
{
  rmm::mr::indexed_pool_memory_resource mr{
    rmm::mr::get_current_device_resource_ref(), 1024, 2048};
  auto* held = mr.allocate_sync(512);
  EXPECT_THROW((void)mr.allocate_sync(2048), rmm::out_of_memory);
  EXPECT_NO_THROW(mr.deallocate_sync(held, 512));
}

TEST(IndexedPoolTest, ReclaimAcrossStreams)
{
  rmm::mr::indexed_pool_memory_resource mr{
    rmm::mr::get_current_device_resource_ref(), 256, 1024};
  rmm::cuda_stream source;
  {
    rmm::device_buffer source_block{256, source.view(), mr};
  }

  auto* ptr = mr.allocate_sync(1024);
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(mr.pool_size(), 1024);
  mr.deallocate_sync(ptr, 1024);
}

TEST(IndexedPoolTest, ReclaimIsStreamOrderedAcrossStreams)
{
  rmm::mr::cuda_async_memory_resource upstream;
  rmm::mr::indexed_pool_memory_resource mr{upstream, 256, 1024};
  host_func_gate prior_work;
  host_func_gate_release_guard const release_prior_work{prior_work};
  rmm::cuda_stream source;

  auto* source_ptr = mr.allocate(source.view(), 256, rmm::CUDA_ALLOCATION_ALIGNMENT);
  RMM_CUDA_TRY(cudaLaunchHostFunc(
    source.value(), [](void* data) { static_cast<host_func_gate*>(data)->wait(); }, &prior_work));
  mr.deallocate(source.view(), source_ptr, 256, rmm::CUDA_ALLOCATION_ALIGNMENT);

  rmm::cuda_stream destination;
  auto* destination_ptr =
    mr.allocate(destination.view(), 1024, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto const reclaim_returned_without_waiting = !prior_work.complete();

  prior_work.release();
  mr.deallocate(
    destination.view(), destination_ptr, 1024, rmm::CUDA_ALLOCATION_ALIGNMENT);
  destination.synchronize();

  EXPECT_TRUE(reclaim_returned_without_waiting);
  EXPECT_TRUE(prior_work.complete());
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

TEST(CoalescingFreeListTest, ReturnsBestFitBelowIndexThreshold)
{
  rmm::mr::detail::coalescing_free_list blocks;
  blocks.insert({reinterpret_cast<char*>(std::uintptr_t{0x10000}), 1024, true});
  blocks.insert({reinterpret_cast<char*>(std::uintptr_t{0x20000}), 512, true});
  blocks.insert({reinterpret_cast<char*>(std::uintptr_t{0x30000}), 768, true});

  auto const result = blocks.get_block(600);
  EXPECT_EQ(result.pointer(), reinterpret_cast<char*>(std::uintptr_t{0x30000}));
  EXPECT_EQ(result.size(), 768);
}

TEST(CoalescingFreeListTest, CoalescesBothNeighbors)
{
  rmm::mr::detail::coalescing_free_list blocks;
  std::array<char, 768> storage{};
  auto* const base = storage.data();
  blocks.insert({base, 256, true});
  blocks.insert(
    {base + 512, 256, false});  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  blocks.insert(
    {base + 256, 256, false});  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

  ASSERT_EQ(blocks.size(), 1);
  auto const result = blocks.get_block(768);
  EXPECT_EQ(result.pointer(), base);
  EXPECT_EQ(result.size(), 768);
  EXPECT_TRUE(result.is_head());
}

TEST(CoalescingFreeListTest, ReturnsBestFitWithLargeBlockCount)
{
  rmm::mr::detail::coalescing_free_list blocks;
  constexpr std::size_t block_count{1100};
  constexpr std::uintptr_t base{0x100000};

  for (std::size_t i = 0; i < block_count; ++i) {
    auto* const ptr = reinterpret_cast<char*>(base + i * 4096);
    blocks.insert({ptr, ((i % 7) + 1) * 256, true});
  }

  auto const result = blocks.get_block(1000);
  EXPECT_EQ(result.size(), 1024);
  EXPECT_EQ(blocks.size(), block_count - 1);
}

TEST(IndexedCoalescingFreeListTest, ReturnsBestFitAfterIndexActivation)
{
  rmm::mr::detail::indexed_coalescing_free_list blocks;
  constexpr std::size_t block_count{1100};
  constexpr std::uintptr_t base{0x200000};

  for (std::size_t i = 0; i < block_count; ++i) {
    auto* const ptr = reinterpret_cast<char*>(base + i * 4096);
    blocks.insert({ptr, ((i % 7) + 1) * 256, true});
  }

  ASSERT_TRUE(blocks.diagnostics_index_active());
  auto const result = blocks.get_block(1000);
  EXPECT_EQ(result.size(), 1024);
  EXPECT_EQ(blocks.size(), block_count - 1);
  EXPECT_TRUE(blocks.diagnostics_indexes_consistent());
}

TEST(IndexedCoalescingFreeListTest, MergeDrainAndReuseAfterIndexActivation)
{
  rmm::mr::detail::indexed_coalescing_free_list destination;
  rmm::mr::detail::indexed_coalescing_free_list source;
  constexpr std::size_t initial_blocks{1100};
  constexpr std::uintptr_t base{0x10000000};

  for (std::size_t i = 0; i < initial_blocks; ++i) {
    destination.insert({reinterpret_cast<char*>(base + i * 8192), 256 + (i % 8) * 256, true});
    source.insert({reinterpret_cast<char*>(base + i * 8192 + 4096), 256 + (i % 7) * 256, true});
  }

  destination.insert(std::move(source));
  ASSERT_EQ(destination.size(), 2 * initial_blocks);

  while (destination.size() > 500) {
    EXPECT_TRUE(destination.get_block(1).is_valid());
  }

  for (std::size_t i = 0; i < 600; ++i) {
    destination.insert(
      {reinterpret_cast<char*>(base + 0x10000000 + i * 4096), 256 + (i % 5) * 256, true});
  }
  ASSERT_EQ(destination.size(), 1100);

  while (!destination.is_empty()) {
    EXPECT_TRUE(destination.get_block(1).is_valid());
  }

  auto* const reused = reinterpret_cast<char*>(base + 0x20000000);
  destination.insert({reused, 4096, true});
  EXPECT_TRUE(destination.diagnostics_indexes_consistent());
  auto const result = destination.get_block(4096);
  EXPECT_EQ(result.pointer(), reused);
  EXPECT_TRUE(destination.is_empty());
}

TEST(IndexedCoalescingFreeListTest, ActiveIndexInsertionMaintainsIndexesForEveryMergeShape)
{
  using free_list = rmm::mr::detail::indexed_coalescing_free_list;
  constexpr std::size_t filler_count{1024};
  constexpr std::uintptr_t filler_base{0x30000000};
  constexpr std::uintptr_t target_base{0x70000000};

  auto activate = [=](free_list& blocks) {
    for (std::size_t index = 0; index < filler_count; ++index) {
      blocks.insert(
        {reinterpret_cast<char*>(filler_base + index * 4096), 128, true});
    }
    ASSERT_TRUE(blocks.diagnostics_index_active());
    ASSERT_TRUE(blocks.diagnostics_indexes_consistent());
  };

  {
    free_list blocks;
    activate(blocks);
    blocks.insert({reinterpret_cast<char*>(target_base), 512, true});
    EXPECT_TRUE(blocks.diagnostics_indexes_consistent());
    EXPECT_EQ(blocks.get_block(512).pointer(), reinterpret_cast<char*>(target_base));
  }
  {
    free_list blocks;
    activate(blocks);
    blocks.insert({reinterpret_cast<char*>(target_base), 256, true});
    blocks.insert({reinterpret_cast<char*>(target_base + 256), 256, false});
    EXPECT_TRUE(blocks.diagnostics_indexes_consistent());
    auto const merged = blocks.get_block(512);
    EXPECT_EQ(merged.pointer(), reinterpret_cast<char*>(target_base));
    EXPECT_EQ(merged.size(), 512);
  }
  {
    free_list blocks;
    activate(blocks);
    blocks.insert({reinterpret_cast<char*>(target_base + 256), 256, false});
    blocks.insert({reinterpret_cast<char*>(target_base), 256, true});
    EXPECT_TRUE(blocks.diagnostics_indexes_consistent());
    auto const merged = blocks.get_block(512);
    EXPECT_EQ(merged.pointer(), reinterpret_cast<char*>(target_base));
    EXPECT_EQ(merged.size(), 512);
  }
  {
    free_list blocks;
    activate(blocks);
    blocks.insert({reinterpret_cast<char*>(target_base), 256, true});
    blocks.insert({reinterpret_cast<char*>(target_base + 512), 256, false});
    blocks.insert({reinterpret_cast<char*>(target_base + 256), 256, false});
    EXPECT_TRUE(blocks.diagnostics_indexes_consistent());
    auto const merged = blocks.get_block(768);
    EXPECT_EQ(merged.pointer(), reinterpret_cast<char*>(target_base));
    EXPECT_EQ(merged.size(), 768);
  }
}

TEST(IndexedCoalescingFreeListTest, ActiveSelectionCommitsSplitAndExactWithoutNewNodes)
{
  using free_list = rmm::mr::detail::indexed_coalescing_free_list;
  constexpr std::size_t filler_count{1024};
  constexpr std::uintptr_t filler_base{0x30000000};
  constexpr std::uintptr_t target_base{0x70000000};
  free_list blocks;

  for (std::size_t index = 0; index < filler_count; ++index) {
    blocks.insert({reinterpret_cast<char*>(filler_base + index * 4096), 128, true});
  }
  blocks.insert({reinterpret_cast<char*>(target_base), 1024, true});
  ASSERT_TRUE(blocks.diagnostics_index_active());

  auto const split = blocks.find_block(768);
  ASSERT_NE(split.block, blocks.end());
  ASSERT_EQ(split.block->pointer(), reinterpret_cast<char*>(target_base));
  blocks.commit_block_selection(
    split, {reinterpret_cast<char*>(target_base + 768), 256, false});
  EXPECT_EQ(blocks.size(), filler_count + 1);
  EXPECT_TRUE(blocks.diagnostics_indexes_consistent());

  auto const exact = blocks.find_block(256);
  ASSERT_NE(exact.block, blocks.end());
  EXPECT_EQ(exact.block->pointer(), reinterpret_cast<char*>(target_base + 768));
  blocks.commit_block_selection(exact, {});
  EXPECT_EQ(blocks.size(), filler_count);
  EXPECT_TRUE(blocks.diagnostics_indexes_consistent());
}

TEST(IndexedCoalescingFreeListTest, FailedLookupCacheTracksRemovalCoalescingAndClear)
{
  rmm::mr::detail::indexed_coalescing_free_list blocks;
  constexpr std::uintptr_t base{0x30000000};

  blocks.insert({reinterpret_cast<char*>(base), 256, true});
  blocks.insert({reinterpret_cast<char*>(base + 4096), 512, true});
  blocks.insert({reinterpret_cast<char*>(base + 8192), 1024, true});
  EXPECT_EQ(blocks.largest_block_size(), 1024);

  // The first oversized miss establishes a cached failure; the repeated miss can use it.
  EXPECT_FALSE(blocks.get_block(2048).is_valid());
  EXPECT_FALSE(blocks.get_block(2048).is_valid());

  // A fitting insertion invalidates the cached failure. After that block is removed, the first miss
  // establishes the failure threshold again and subsequent misses must remain correct.
  blocks.insert({reinterpret_cast<char*>(base + 12288), 4096, true});
  EXPECT_EQ(blocks.largest_block_size(), 4096);
  EXPECT_EQ(blocks.get_block(2048).size(), 4096);
  EXPECT_EQ(blocks.largest_block_size(), 1024);
  EXPECT_FALSE(blocks.get_block(2048).is_valid());
  EXPECT_FALSE(blocks.get_block(2048).is_valid());

  // An insertion that coalesces into a fitting block must invalidate the cached failure.
  blocks.insert({reinterpret_cast<char*>(base + 256), 2048, false});
  auto const coalesced = blocks.get_block(2048);
  EXPECT_EQ(coalesced.pointer(), reinterpret_cast<char*>(base));
  EXPECT_EQ(coalesced.size(), 2304);

  blocks.clear();
  blocks.insert({reinterpret_cast<char*>(base), 256, true});
  EXPECT_FALSE(blocks.get_block(512).is_valid());
}

TEST(IndexedCoalescingFreeListTest, ExactExtractionPreservesActiveIndexes)
{
  using free_list = rmm::mr::detail::indexed_coalescing_free_list;
  constexpr std::uintptr_t source_base{0x50000000};
  constexpr std::uintptr_t destination_base{0x70000000};
  free_list source;
  free_list extracted;
  free_list destination;

  for (std::size_t index = 0; index < 1100; ++index) {
    source.insert({reinterpret_cast<char*>(source_base + index * 4096), 256, true});
  }
  for (std::size_t index = 0; index < 1023; ++index) {
    destination.insert({reinterpret_cast<char*>(destination_base + index * 4096), 256, true});
  }

  ASSERT_TRUE(source.diagnostics_index_active());
  auto selected = source.begin();
  std::advance(selected, 500);
  source.extract_exact(selected, extracted);
  EXPECT_EQ(source.size(), 1099);
  EXPECT_EQ(extracted.size(), 1);
  EXPECT_TRUE(source.diagnostics_indexes_consistent());

  destination.prepare_for_splice(1);
  ASSERT_TRUE(destination.diagnostics_index_active());
  auto prepared = destination.prepare_splice(extracted.begin());
  destination.commit_prepared_splice(extracted, extracted.begin(), std::move(prepared));
  EXPECT_EQ(destination.size(), 1024);
  EXPECT_TRUE(extracted.is_empty());
  EXPECT_TRUE(destination.diagnostics_indexes_consistent());
}

namespace {
int injected_wait_calls{};

cudaError_t fail_cross_stream_wait(cudaStream_t, cudaEvent_t, unsigned int)
{
  ++injected_wait_calls;
  return cudaErrorInvalidValue;
}

cudaError_t fail_second_recovery_wait(cudaStream_t, cudaEvent_t, unsigned int)
{
  ++injected_wait_calls;
  return injected_wait_calls == 2 ? cudaErrorInvalidValue : cudaSuccess;
}

cudaError_t count_recovery_wait(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
  ++injected_wait_calls;
  return cudaStreamWaitEvent(stream, event, flags);
}

cudaError_t fail_recovery_record(cudaEvent_t, cudaStream_t) { return cudaErrorInvalidValue; }

struct publication_operation {
  bool is_record{};
  cudaStream_t stream{};
  cudaEvent_t event{};
};

std::vector<publication_operation> publication_operations;
std::vector<cudaEvent_t> pending_publication_dependencies;
std::vector<cudaEvent_t> captured_publication_dependencies;
cudaEvent_t captured_publication_event{};

cudaError_t model_recovery_wait(cudaStream_t stream, cudaEvent_t event, unsigned int)
{
  publication_operations.push_back({false, stream, event});
  if (std::find(pending_publication_dependencies.cbegin(),
                pending_publication_dependencies.cend(),
                event) == pending_publication_dependencies.cend()) {
    pending_publication_dependencies.push_back(event);
  }
  return cudaSuccess;
}

cudaError_t model_recovery_record(cudaEvent_t event, cudaStream_t stream)
{
  publication_operations.push_back({true, stream, event});
  captured_publication_event        = event;
  captured_publication_dependencies = pending_publication_dependencies;
  return cudaSuccess;
}
}  // namespace

TEST(IndexedPoolMemoryResourceTest, CrossStreamWaitFailurePreservesSingleDonorBlock)
{
  constexpr std::size_t block_size{rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t pool_size{2 * block_size};
  using hooks = rmm::mr::detail::indexed_recovery_test_hooks;

  rmm::mr::indexed_pool_memory_resource pool{
    rmm::mr::get_current_device_resource_ref(), pool_size, pool_size};
  rmm::device_async_resource_ref resource{pool};
  rmm::cuda_stream owner;
  rmm::cuda_stream requester;

  auto* donor = resource.allocate(requester.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto* held  = resource.allocate(requester.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(owner.view(), donor, block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  injected_wait_calls = 0;
  hooks::wait         = fail_cross_stream_wait;
  EXPECT_THROW(
    (void)resource.allocate(requester.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT),
    rmm::cuda_error);
  EXPECT_EQ(injected_wait_calls, 1);
  hooks::reset();

  auto* recovered =
    resource.allocate(requester.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_EQ(recovered, donor);
  resource.deallocate(requester.view(), recovered, block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(requester.view(), held, block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

TEST(IndexedPoolMemoryResourceTest, ActiveIndexWaitFailurePreservesSingleDonorBlock)
{
  constexpr std::size_t block_size{rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t owner_count{25};
  constexpr std::size_t pool_size{owner_count * block_size};
  using hooks = rmm::mr::detail::indexed_recovery_test_hooks;

  rmm::mr::indexed_pool_memory_resource pool{
    rmm::mr::get_current_device_resource_ref(), pool_size, pool_size};
  rmm::device_async_resource_ref resource{pool};
  std::vector<rmm::cuda_stream> owners(owner_count);
  std::vector<void*> blocks(owner_count);
  for (auto& block : blocks) {
    block = resource.allocate(rmm::cuda_stream_legacy, block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  for (std::size_t index = 0; index < owner_count; ++index) {
    resource.deallocate(
      owners[index].view(), blocks[index], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }

  rmm::cuda_stream requester;
  injected_wait_calls = 0;
  hooks::wait         = fail_cross_stream_wait;
  EXPECT_THROW(
    (void)resource.allocate(requester.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT),
    rmm::cuda_error);
  EXPECT_EQ(injected_wait_calls, 1);
  hooks::reset();

  std::vector<void*> recovered;
  recovered.reserve(owner_count);
  for (std::size_t index = 0; index < owner_count; ++index) {
    auto* ptr = resource.allocate(requester.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    EXPECT_NE(std::find(blocks.cbegin(), blocks.cend(), ptr), blocks.cend());
    EXPECT_EQ(std::count(recovered.cbegin(), recovered.cend(), ptr), 0);
    recovered.push_back(ptr);
  }
  for (auto* ptr : recovered) {
    resource.deallocate(requester.view(), ptr, block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
}

TEST(IndexedPoolMemoryResourceTest, ThreeStreamSelectiveRecoveryPublishesRemainder)
{
  constexpr std::size_t chunk_size{2 * rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t suffix_size{rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t request_size{chunk_size + suffix_size};
  constexpr std::size_t pool_size{3 * chunk_size};
  rmm::mr::indexed_pool_memory_resource pool{
    rmm::mr::get_current_device_resource_ref(), pool_size, pool_size};
  rmm::device_async_resource_ref resource{pool};
  rmm::cuda_stream owner_a;
  rmm::cuda_stream owner_b;
  rmm::cuda_stream owner_c;

  std::array<void*, 3> blocks{};
  blocks[0] = resource.allocate(owner_b.view(), chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  blocks[1] = resource.allocate(owner_a.view(), chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  blocks[2] = resource.allocate(owner_c.view(), chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  auto* const expected_suffix = static_cast<char*>(blocks[1]) + suffix_size;
  RMM_CUDA_TRY(cudaMemsetAsync(expected_suffix, 0x5a, suffix_size, owner_a.value()));
  resource.deallocate(owner_b.view(), blocks[0], chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(owner_a.view(), blocks[1], chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  auto* recovered = resource.allocate(owner_b.view(), request_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto* remainder = resource.allocate(owner_c.view(), suffix_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_EQ(recovered, blocks[0]);
  EXPECT_EQ(remainder, expected_suffix);

  std::array<unsigned char, suffix_size> host{};
  RMM_CUDA_TRY(
    cudaMemcpyAsync(host.data(), remainder, suffix_size, cudaMemcpyDeviceToHost, owner_c.value()));
  owner_c.synchronize();
  EXPECT_TRUE(std::all_of(host.cbegin(), host.cend(), [](auto value) { return value == 0x5a; }));

  resource.deallocate(owner_b.view(), recovered, request_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(owner_c.view(), remainder, suffix_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(owner_b.view(), blocks[2], chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

TEST(IndexedPoolMemoryResourceTest, RecoveryPublishesAllUniqueDonorsBeforeCommit)
{
  constexpr std::size_t block_size{rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t pool_size{4 * block_size};
  using hooks = rmm::mr::detail::indexed_recovery_test_hooks;

  rmm::mr::indexed_pool_memory_resource pool{
    rmm::mr::get_current_device_resource_ref(), pool_size, pool_size};
  rmm::device_async_resource_ref resource{pool};
  rmm::cuda_stream owner_a;
  rmm::cuda_stream owner_b;
  rmm::cuda_stream requester;
  std::array<void*, 4> blocks{};
  for (auto& block : blocks) {
    block = resource.allocate(requester.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }

  resource.deallocate(owner_a.view(), blocks[0], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(owner_b.view(), blocks[1], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(owner_a.view(), blocks[2], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  publication_operations.clear();
  pending_publication_dependencies.clear();
  captured_publication_dependencies.clear();
  captured_publication_event = nullptr;
  hooks::wait                = model_recovery_wait;
  hooks::record              = model_recovery_record;

  void* recovered{};
  try {
    recovered = resource.allocate(requester.view(), 3 * block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  } catch (...) {
    hooks::reset();
    throw;
  }
  hooks::reset();

  ASSERT_EQ(publication_operations.size(), 3);
  EXPECT_FALSE(publication_operations[0].is_record);
  EXPECT_FALSE(publication_operations[1].is_record);
  EXPECT_TRUE(publication_operations[2].is_record);
  EXPECT_EQ(publication_operations[0].stream, requester.value());
  EXPECT_EQ(publication_operations[1].stream, requester.value());
  EXPECT_EQ(publication_operations[2].stream, requester.value());
  EXPECT_NE(publication_operations[0].event, publication_operations[1].event);
  EXPECT_EQ(captured_publication_event, publication_operations[2].event);
  ASSERT_EQ(captured_publication_dependencies.size(), 2);
  EXPECT_EQ(captured_publication_dependencies[0], publication_operations[0].event);
  EXPECT_EQ(captured_publication_dependencies[1], publication_operations[1].event);

  resource.deallocate(requester.view(), recovered, 3 * block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(requester.view(), blocks[3], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

TEST(IndexedPoolMemoryResourceTest, WaitAndRecordFailuresDoNotMutateOwnership)
{
  constexpr std::size_t block_size{rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t pool_size{4 * block_size};
  using hooks = rmm::mr::detail::indexed_recovery_test_hooks;

  auto run_failure = [=](bool fail_record) {
    rmm::mr::indexed_pool_memory_resource pool{
      rmm::mr::get_current_device_resource_ref(), pool_size, pool_size};
    rmm::device_async_resource_ref resource{pool};
    rmm::cuda_stream owner_a;
    rmm::cuda_stream owner_b;
    rmm::cuda_stream owner_c;
    rmm::cuda_stream requester;
    std::array<void*, 4> blocks{};
    for (auto& block : blocks) {
      block = resource.allocate(requester.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    }
    resource.deallocate(owner_a.view(), blocks[0], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    resource.deallocate(owner_b.view(), blocks[1], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    resource.deallocate(owner_c.view(), blocks[2], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

    injected_wait_calls = 0;
    hooks::wait         = fail_record ? count_recovery_wait : fail_second_recovery_wait;
    hooks::record       = fail_record ? fail_recovery_record : hooks::default_record;
    EXPECT_THROW(
      (void)resource.allocate(requester.view(), 3 * block_size, rmm::CUDA_ALLOCATION_ALIGNMENT),
      rmm::cuda_error);
    hooks::reset();

    auto* recovered =
      resource.allocate(requester.view(), 3 * block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    EXPECT_EQ(recovered, blocks[0]);
    resource.deallocate(
      requester.view(), recovered, 3 * block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    resource.deallocate(requester.view(), blocks[3], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  };

  run_failure(false);
  run_failure(true);
}

TEST(IndexedPoolMemoryResourceTest, MetadataFailuresPrecedePublicationAndPreserveOwnership)
{
  constexpr std::size_t chunk_size{2 * rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t request_size{3 * rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t pool_size{3 * chunk_size};
  using hooks = rmm::mr::detail::indexed_recovery_test_hooks;

  // A partial allocation from two contiguous chunks exercises every recovery staging checkpoint,
  // including preparation of the split remainder. Advance the failure point until the first
  // successful attempt instead of coupling the test to an exact checkpoint count.
  bool reached_success{};
  for (int failure_point = 0; failure_point < 32 && !reached_success; ++failure_point) {
    rmm::mr::indexed_pool_memory_resource pool{
      rmm::mr::get_current_device_resource_ref(), pool_size, pool_size};
    rmm::device_async_resource_ref resource{pool};
    rmm::cuda_stream owner_a;
    rmm::cuda_stream owner_b;
    rmm::cuda_stream requester;
    std::array<void*, 3> blocks{};
    for (auto& block : blocks) {
      block = resource.allocate(requester.view(), chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    }
    resource.deallocate(owner_a.view(), blocks[0], chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    resource.deallocate(owner_b.view(), blocks[1], chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

    injected_wait_calls        = 0;
    hooks::metadata_fail_after = failure_point;
    hooks::wait                = count_recovery_wait;

    void* recovered{};
    bool metadata_failed{};
    try {
      recovered =
        resource.allocate(requester.view(), request_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    } catch (std::bad_alloc const&) {
      metadata_failed = true;
    }
    hooks::reset();

    if (metadata_failed) {
      EXPECT_EQ(injected_wait_calls, 0);
      recovered =
        resource.allocate(requester.view(), request_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    } else {
      reached_success = true;
      EXPECT_GT(injected_wait_calls, 0);
    }

    EXPECT_EQ(recovered, blocks[0]);
    resource.deallocate(requester.view(), recovered, request_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    resource.deallocate(requester.view(), blocks[2], chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  EXPECT_TRUE(reached_success);
}

TEST(IndexedPoolMemoryResourceTest, FailedRecoveryCacheInvalidatesOnFree)
{
  constexpr std::size_t block_size{rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t pool_size{4 * block_size};
  rmm::mr::indexed_pool_memory_resource pool{
    rmm::mr::get_current_device_resource_ref(), pool_size, pool_size};
  rmm::device_async_resource_ref resource{pool};
  rmm::cuda_stream owner_a;
  rmm::cuda_stream owner_b;
  rmm::cuda_stream requester;
  std::array<void*, 4> blocks{};
  for (auto& block : blocks) {
    block = resource.allocate(requester.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }

  resource.deallocate(owner_a.view(), blocks[0], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(owner_b.view(), blocks[2], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  EXPECT_THROW(
    (void)resource.allocate(requester.view(), 2 * block_size, rmm::CUDA_ALLOCATION_ALIGNMENT),
    rmm::out_of_memory);
  EXPECT_THROW(
    (void)resource.allocate(requester.view(), 2 * block_size, rmm::CUDA_ALLOCATION_ALIGNMENT),
    rmm::out_of_memory);

  resource.deallocate(requester.view(), blocks[1], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto* recovered =
    resource.allocate(requester.view(), 2 * block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_EQ(recovered, blocks[0]);

  resource.deallocate(requester.view(), recovered, 2 * block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(requester.view(), blocks[3], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

TEST(IndexedPoolMemoryResourceTest, MultipleSelectedNodesFromOneDonorRemainValid)
{
  constexpr std::size_t block_size{rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t pool_size{4 * block_size};
  rmm::mr::indexed_pool_memory_resource pool{
    rmm::mr::get_current_device_resource_ref(), pool_size, pool_size};
  rmm::device_async_resource_ref resource{pool};
  rmm::cuda_stream owner_a;
  rmm::cuda_stream owner_b;
  rmm::cuda_stream requester;
  std::array<void*, 4> blocks{};
  for (auto& block : blocks) {
    block = resource.allocate(requester.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  resource.deallocate(owner_a.view(), blocks[0], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(owner_b.view(), blocks[1], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(owner_a.view(), blocks[2], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(owner_b.view(), blocks[3], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  auto* recovered = resource.allocate(requester.view(), pool_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_EQ(recovered, blocks[0]);
  resource.deallocate(requester.view(), recovered, pool_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

TEST(IndexedPoolMemoryResourceTest, RecoverySkipsLongInsufficientPrefix)
{
  constexpr std::size_t block_size{rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t prefix_blocks{16};
  constexpr std::size_t request_blocks{17};
  constexpr std::size_t gap_index{prefix_blocks};
  constexpr std::size_t suffix_index{gap_index + 1};
  constexpr std::size_t tail_index{suffix_index + request_blocks};
  constexpr std::size_t block_count{tail_index + 1};
  constexpr std::size_t pool_size{block_count * block_size};

  rmm::mr::indexed_pool_memory_resource pool{
    rmm::mr::get_current_device_resource_ref(), pool_size, pool_size};
  rmm::device_async_resource_ref resource{pool};
  rmm::cuda_stream owner_a;
  rmm::cuda_stream owner_b;
  rmm::cuda_stream requester;
  std::array<void*, block_count> blocks{};
  for (auto& block : blocks) {
    block = resource.allocate(requester.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }

  auto free_to_alternating_owner = [&](std::size_t index) {
    auto const owner = (index % 2 == 0) ? owner_a.view() : owner_b.view();
    resource.deallocate(owner, blocks[index], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  };
  for (std::size_t index = 0; index < prefix_blocks; ++index) {
    free_to_alternating_owner(index);
  }
  for (std::size_t index = suffix_index; index < tail_index; ++index) {
    free_to_alternating_owner(index);
  }

  auto* recovered = resource.allocate(
    requester.view(), request_blocks * block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_EQ(recovered, blocks[suffix_index]);

  std::vector<void*> recovered_prefix(prefix_blocks);
  for (auto& block : recovered_prefix) {
    block = resource.allocate(requester.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  for (std::size_t index = 0; index < prefix_blocks; ++index) {
    EXPECT_EQ(std::count(recovered_prefix.cbegin(), recovered_prefix.cend(), blocks[index]), 1);
  }

  resource.deallocate(
    requester.view(), recovered, request_blocks * block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  for (auto* block : recovered_prefix) {
    resource.deallocate(requester.view(), block, block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  resource.deallocate(
    requester.view(), blocks[gap_index], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(
    requester.view(), blocks[tail_index], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

TEST(IndexedPoolMemoryResourceTest, PerThreadDefaultThreeThreadTransitivePublication)
{
  constexpr std::size_t chunk_size{2 * rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t suffix_size{rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t request_size{chunk_size + suffix_size};
  constexpr std::size_t pool_size{3 * chunk_size};
  rmm::mr::indexed_pool_memory_resource pool{
    rmm::mr::get_current_device_resource_ref(), pool_size, pool_size};
  rmm::device_async_resource_ref resource{pool};
  std::array<void*, 3> blocks{};
  for (auto& block : blocks) {
    block = pool.allocate_sync(chunk_size);
  }

  std::atomic<int> stage{};
  void* recovered{};
  void* remainder{};
  std::array<unsigned char, suffix_size> host{};
  std::array<std::exception_ptr, 3> failures{};

  std::thread owner_a{[&] {
    try {
      auto const stream           = cuda::stream_ref{rmm::cuda_stream_per_thread.value()};
      auto* const expected_suffix = static_cast<char*>(blocks[1]) + suffix_size;
      RMM_CUDA_TRY(
        cudaMemsetAsync(expected_suffix, 0x6b, suffix_size, rmm::cuda_stream_per_thread.value()));
      resource.deallocate(stream, blocks[1], chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
      stage.store(1, std::memory_order_release);
    } catch (...) {
      failures[0] = std::current_exception();
      stage.store(4, std::memory_order_release);
    }
  }};
  std::thread owner_b{[&] {
    while (stage.load(std::memory_order_acquire) < 1) {
      std::this_thread::yield();
    }
    if (stage.load(std::memory_order_acquire) >= 4) { return; }
    try {
      auto const stream = cuda::stream_ref{rmm::cuda_stream_per_thread.value()};
      resource.deallocate(stream, blocks[0], chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
      recovered = resource.allocate(stream, request_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
      stage.store(2, std::memory_order_release);
    } catch (...) {
      failures[1] = std::current_exception();
      stage.store(4, std::memory_order_release);
    }
  }};
  std::thread owner_c{[&] {
    while (stage.load(std::memory_order_acquire) < 2) {
      std::this_thread::yield();
    }
    if (stage.load(std::memory_order_acquire) >= 4) { return; }
    try {
      auto const stream = cuda::stream_ref{rmm::cuda_stream_per_thread.value()};
      remainder         = resource.allocate(stream, suffix_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
      RMM_CUDA_TRY(cudaMemcpyAsync(host.data(),
                                   remainder,
                                   suffix_size,
                                   cudaMemcpyDeviceToHost,
                                   rmm::cuda_stream_per_thread.value()));
      rmm::cuda_stream_per_thread.synchronize();
      stage.store(3, std::memory_order_release);
    } catch (...) {
      failures[2] = std::current_exception();
      stage.store(4, std::memory_order_release);
    }
  }};
  owner_a.join();
  owner_b.join();
  owner_c.join();

  for (auto const& failure : failures) {
    if (failure) { std::rethrow_exception(failure); }
  }
  EXPECT_EQ(recovered, blocks[0]);
  EXPECT_EQ(remainder, static_cast<char*>(blocks[1]) + suffix_size);
  EXPECT_TRUE(std::all_of(host.cbegin(), host.cend(), [](auto value) { return value == 0x6b; }));

  pool.deallocate_sync(recovered, request_size);
  pool.deallocate_sync(remainder, suffix_size);
  pool.deallocate_sync(blocks[2], chunk_size);
}

TEST(IndexedPoolMemoryResourceTest, RepeatedAlternatingEventGenerations)
{
  constexpr std::size_t chunk_size{2 * rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t partial_size{3 * rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t region_size{2 * chunk_size};
  rmm::mr::indexed_pool_memory_resource pool{
    rmm::mr::get_current_device_resource_ref(), 3 * chunk_size, 3 * chunk_size};
  rmm::device_async_resource_ref resource{pool};
  rmm::cuda_stream owner_a;
  rmm::cuda_stream owner_b;
  std::array<void*, 3> blocks{};
  for (auto& block : blocks) {
    block = resource.allocate(owner_b.view(), chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  resource.deallocate(owner_b.view(), blocks[0], chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(owner_a.view(), blocks[1], chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  for (int generation = 0; generation < 8; ++generation) {
    auto* partial = resource.allocate(owner_b.view(), partial_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    RMM_CUDA_TRY(cudaMemsetAsync(partial, generation + 1, partial_size, owner_b.value()));
    resource.deallocate(owner_a.view(), partial, partial_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

    auto* whole = resource.allocate(owner_a.view(), region_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    std::array<unsigned char, partial_size> host{};
    RMM_CUDA_TRY(
      cudaMemcpyAsync(host.data(), whole, partial_size, cudaMemcpyDeviceToHost, owner_a.value()));
    owner_a.synchronize();
    EXPECT_TRUE(std::all_of(host.cbegin(), host.cend(), [generation](auto value) {
      return value == static_cast<unsigned char>(generation + 1);
    }));

    resource.deallocate(owner_b.view(), whole, region_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    auto* first  = resource.allocate(owner_b.view(), chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    auto* second = resource.allocate(owner_b.view(), chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    resource.deallocate(owner_b.view(), first, chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    resource.deallocate(owner_a.view(), second, chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }

  auto* final_region =
    resource.allocate(owner_b.view(), region_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(owner_b.view(), final_region, region_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(owner_b.view(), blocks[2], chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

TEST(IndexedPoolMemoryResourceTest, DeletedDonorStreamSelectiveRecovery)
{
  constexpr std::size_t chunk_size{2 * rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t request_size{3 * rmm::CUDA_ALLOCATION_ALIGNMENT};
  rmm::mr::indexed_pool_memory_resource pool{
    rmm::mr::get_current_device_resource_ref(), 3 * chunk_size, 3 * chunk_size};
  rmm::device_async_resource_ref resource{pool};
  rmm::cuda_stream requester;
  std::array<void*, 3> blocks{};
  for (auto& block : blocks) {
    block = resource.allocate(requester.view(), chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  resource.deallocate(requester.view(), blocks[0], chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  cudaStream_t donor{};
  RMM_CUDA_TRY(cudaStreamCreate(&donor));
  RMM_CUDA_TRY(cudaMemsetAsync(blocks[1], 0x4d, chunk_size, donor));
  resource.deallocate(
    cuda::stream_ref{donor}, blocks[1], chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  RMM_CUDA_TRY(cudaStreamDestroy(donor));

  auto* recovered =
    resource.allocate(requester.view(), request_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  requester.synchronize();
  EXPECT_EQ(recovered, blocks[0]);
  resource.deallocate(requester.view(), recovered, request_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(requester.view(), blocks[2], chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

TEST(IndexedPoolMemoryResourceTest, RecoveryRemainderCoalescesWithRequesterNeighbor)
{
  constexpr std::size_t chunk_size{2 * rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t request_size{3 * rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t pool_size{4 * chunk_size};
  rmm::mr::indexed_pool_memory_resource pool{
    rmm::mr::get_current_device_resource_ref(), pool_size, pool_size};
  rmm::device_async_resource_ref resource{pool};
  rmm::cuda_stream donor;
  rmm::cuda_stream requester;
  std::array<void*, 4> blocks{};
  for (auto& block : blocks) {
    block = resource.allocate(requester.view(), chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }

  resource.deallocate(requester.view(), blocks[0], chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(donor.view(), blocks[1], chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(requester.view(), blocks[2], chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  auto* recovered =
    resource.allocate(requester.view(), request_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto* coalesced =
    resource.allocate(requester.view(), request_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_EQ(recovered, blocks[0]);
  EXPECT_EQ(coalesced, static_cast<char*>(blocks[1]) + rmm::CUDA_ALLOCATION_ALIGNMENT);

  resource.deallocate(requester.view(), recovered, request_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(requester.view(), coalesced, request_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(requester.view(), blocks[3], chunk_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

TEST(IndexedPoolMemoryResourceTest, AllocateDeallocateAndSharedOwnership)
{
  rmm::mr::indexed_pool_memory_resource pool{rmm::mr::get_current_device_resource_ref(),
                                             1024 * 1024};
  auto copy = pool;

  constexpr std::size_t size{4096};
  auto* ptr = pool.allocate_sync(size);
  ASSERT_NE(ptr, nullptr);
  EXPECT_NO_THROW(copy.deallocate_sync(ptr, size));
  EXPECT_GE(pool.pool_size(), 1024 * 1024);
}

TEST(IndexedPoolMemoryResourceTest, Equality)
{
  rmm::mr::indexed_pool_memory_resource pool{rmm::mr::get_current_device_resource_ref(),
                                             1024 * 1024};
  auto copy = pool;
  EXPECT_EQ(pool, copy);

  rmm::mr::indexed_pool_memory_resource other{rmm::mr::get_current_device_resource_ref(),
                                              1024 * 1024};
  EXPECT_NE(pool, other);
}

TEST(IndexedPoolMemoryResourceTest, SharedOwnerIndexReusesEmptyOwnerEntries)
{
  constexpr std::size_t block_size{rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t owner_count{25};
  constexpr std::size_t pool_size{owner_count * block_size};

  rmm::mr::indexed_pool_memory_resource pool{
    rmm::mr::get_current_device_resource_ref(), pool_size, pool_size};
  rmm::device_async_resource_ref resource{pool};

  std::vector<rmm::cuda_stream> owners(owner_count);
  std::vector<void*> blocks;
  blocks.reserve(owner_count);
  for (std::size_t i = 0; i < owner_count; ++i) {
    blocks.push_back(
      resource.allocate(rmm::cuda_stream_legacy, block_size, rmm::CUDA_ALLOCATION_ALIGNMENT));
  }
  for (std::size_t i = 0; i < owner_count; ++i) {
    resource.deallocate(owners[i].view(), blocks[i], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }

  rmm::cuda_stream requester;
  cudaEvent_t handoff{};
  RMM_CUDA_TRY(cudaEventCreateWithFlags(&handoff, cudaEventDisableTiming));
  for (std::size_t i = 0; i < owner_count; ++i) {
    auto* ptr = resource.allocate(requester.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    ASSERT_NE(ptr, nullptr);

    RMM_CUDA_TRY(cudaEventRecord(handoff, requester.value()));
    RMM_CUDA_TRY(cudaStreamWaitEvent(owners[0].value(), handoff, 0));
    resource.deallocate(owners[0].view(), ptr, block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  owners[0].synchronize();
  requester.synchronize();
  RMM_CUDA_TRY(cudaEventDestroy(handoff));
}

TEST(IndexedPoolMemoryResourceTest, FreshOwnerSelectiveRecoveryWithActiveMaximumIndex)
{
  constexpr std::size_t block_size{rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t owner_count{25};
  constexpr std::size_t pool_size{owner_count * block_size};

  rmm::mr::indexed_pool_memory_resource pool{
    rmm::mr::get_current_device_resource_ref(), pool_size, pool_size};
  rmm::device_async_resource_ref resource{pool};
  std::vector<rmm::cuda_stream> owners(owner_count);
  std::vector<void*> blocks(owner_count);
  for (auto& block : blocks) {
    block = resource.allocate(rmm::cuda_stream_legacy, block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  for (std::size_t i = 0; i < owner_count; ++i) {
    resource.deallocate(owners[i].view(), blocks[i], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }

  // This requester is created after the owner-maximum index activates. Its empty index entry must
  // exist before selective recovery commits the first two contiguous blocks to it.
  rmm::cuda_stream requester;
  auto* recovered =
    resource.allocate(requester.view(), 2 * block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_EQ(recovered, blocks.front());

  // Recovery changes only the first two owners. Their maximum entries must no longer advertise
  // consumed blocks, while every unaffected active-index entry remains usable.
  std::vector<void*> unaffected;
  unaffected.reserve(owner_count - 2);
  for (std::size_t index = 2; index < owner_count; ++index) {
    auto* ptr = resource.allocate(requester.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    EXPECT_NE(std::find(blocks.cbegin() + 2, blocks.cend(), ptr), blocks.cend());
    EXPECT_EQ(std::count(unaffected.cbegin(), unaffected.cend(), ptr), 0);
    unaffected.push_back(ptr);
  }

  resource.deallocate(requester.view(), recovered, 2 * block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  for (auto* ptr : unaffected) {
    resource.deallocate(requester.view(), ptr, block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
}

TEST(IndexedPoolMemoryResourceTest, RecoveryPublishesDonorDependenciesToThirdStream)
{
  constexpr std::size_t block_size{rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t pool_size{4 * block_size};
  rmm::mr::indexed_pool_memory_resource pool{
    rmm::mr::get_current_device_resource_ref(), pool_size, pool_size};
  rmm::device_async_resource_ref resource{pool};

  rmm::cuda_stream owner_a;
  rmm::cuda_stream owner_b;
  rmm::cuda_stream requester;
  rmm::cuda_stream third_stream;

  std::array<void*, 4> blocks{};
  for (auto& block : blocks) {
    block = resource.allocate(owner_a.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }

  resource.deallocate(owner_a.view(), blocks[0], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(owner_a.view(), blocks[1], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  std::atomic<bool> release_owner_b{false};
  struct release_guard {
    std::atomic<bool>& flag;
    ~release_guard() { flag.store(true, std::memory_order_release); }
  } guard{release_owner_b};

  RMM_CUDA_TRY(cudaLaunchHostFunc(
    owner_b.value(),
    [](void* flag) {
      auto* release = static_cast<std::atomic<bool>*>(flag);
      while (!release->load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
    },
    &release_owner_b));

  resource.deallocate(owner_b.view(), blocks[2], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(owner_b.view(), blocks[3], block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  // Neither owner's 512-byte list can satisfy this request. Recovery waits on both owners,
  // publishes those waits through requester's event, coalesces, and leaves a 256-byte remainder.
  auto* recovered =
    resource.allocate(requester.view(), 3 * block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto* remainder =
    resource.allocate(third_stream.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  cudaEvent_t third_done{};
  RMM_CUDA_TRY(cudaEventCreateWithFlags(&third_done, cudaEventDisableTiming));
  RMM_CUDA_TRY(cudaEventRecord(third_done, third_stream.value()));
  EXPECT_EQ(cudaErrorNotReady, cudaEventQuery(third_done));

  release_owner_b.store(true, std::memory_order_release);
  RMM_CUDA_TRY(cudaEventSynchronize(third_done));
  RMM_CUDA_TRY(cudaEventDestroy(third_done));

  resource.deallocate(requester.view(), recovered, 3 * block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(third_stream.view(), remainder, block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

TEST(IndexedPoolMemoryResourceTest, ExpansionPublishesRemainderReadiness)
{
  constexpr std::size_t block_size{rmm::CUDA_ALLOCATION_ALIGNMENT};
  auto release_upstream = std::make_shared<std::atomic<bool>>(false);
  struct release_guard {
    std::shared_ptr<std::atomic<bool>> flag;
    ~release_guard() { flag->store(true, std::memory_order_release); }
  } guard{release_upstream};

  delayed_async_memory_resource upstream{release_upstream};
  rmm::mr::indexed_pool_memory_resource pool{upstream, 0, 4 * block_size};
  rmm::device_async_resource_ref resource{pool};
  rmm::cuda_stream requester;
  rmm::cuda_stream third_stream;

  // The 256-byte request grows the empty pool by 512 bytes. The delayed upstream resource makes
  // that block ready only after its callback, so the globally visible remainder must carry an
  // event recorded after the callback.
  auto* requested = resource.allocate(requester.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto* remainder =
    resource.allocate(third_stream.view(), block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);

  cudaEvent_t third_done{};
  RMM_CUDA_TRY(cudaEventCreateWithFlags(&third_done, cudaEventDisableTiming));
  RMM_CUDA_TRY(cudaEventRecord(third_done, third_stream.value()));
  EXPECT_EQ(cudaErrorNotReady, cudaEventQuery(third_done));

  release_upstream->store(true, std::memory_order_release);
  RMM_CUDA_TRY(cudaEventSynchronize(third_done));
  RMM_CUDA_TRY(cudaEventDestroy(third_done));

  resource.deallocate(requester.view(), requested, block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  resource.deallocate(third_stream.view(), remainder, block_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
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
static_assert(
  cuda::mr::resource_with<rmm::mr::indexed_pool_memory_resource, cuda::mr::device_accessible>);

}  // namespace test_properties

}  // namespace rmm::test
