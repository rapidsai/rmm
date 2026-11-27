/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../byte_literals.hpp"
#include "../mock_resource.hpp"
#include "test_utils.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/fixed_slab_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>

namespace rmm::test {
namespace {

using fixed_slab_mr = rmm::mr::fixed_slab_memory_resource;
using statistics_mr = rmm::mr::statistics_resource_adaptor;

class FixedSlabMemoryResourceTest : public ::testing::Test {};

// ---------------------------------------------------------------------------
// Construction validation: no CUDA calls may happen for invalid sizes (plain TEST)
// ---------------------------------------------------------------------------

TEST(FixedSlabConstruction, InvalidSlabSize)
{
  mock_resource upstream;
  mock_resource_wrapper wrapper{&upstream};
  // Strict: no allocation/free may be attempted for an invalid slab size.
  EXPECT_CALL(upstream, allocate).Times(0);
  EXPECT_CALL(upstream, deallocate).Times(0);

  EXPECT_THROW(((void)fixed_slab_mr{wrapper, cuda::stream_ref{cudaStreamLegacy}, 0}),
               rmm::logic_error);
  EXPECT_THROW(((void)fixed_slab_mr{wrapper, cuda::stream_ref{cudaStreamLegacy}, 255}),
               rmm::logic_error);
  EXPECT_THROW(((void)fixed_slab_mr{wrapper, cuda::stream_ref{cudaStreamLegacy}, 257}),
               rmm::logic_error);
}

// Host callback that spins until a host flag is set; used to block a stream so a no-wait
// carve can be proven to return. The test's release_on_exit guard unblocks it during
// unwinding, before the resource destructor can synchronize behind the pending callback.
void CUDART_CB spin_until_released(void* data)
{
  auto* released = static_cast<std::atomic<bool>*>(data);
  while (!released->load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
}

// Sets the flag on scope exit so the stream callback can never block teardown, then waits for
// the stream so the callback has exited before teardown continues. Owns the flag on the heap:
// if the synchronize fails, callback completion is unconfirmed, so the flag is deliberately
// leaked rather than freed while a pending callback could still read it.
struct release_on_exit {
  std::unique_ptr<std::atomic<bool>> flag;
  cudaStream_t stream;

  explicit release_on_exit(cudaStream_t stream)
    : flag{std::make_unique<std::atomic<bool>>(false)}, stream{stream}
  {
  }

  ~release_on_exit()
  {
    flag->store(true, std::memory_order_release);
    // The flag is already true, so the callback exits immediately and this cannot hang.
    cudaError_t const status = cudaStreamSynchronize(stream);
    if (status == cudaSuccess) { return; }
    // Synchronize failed: report without throwing (destructors must not throw), and keep the
    // flag alive for a callback that may still be spinning.
    ADD_FAILURE() << "cudaStreamSynchronize failed: " << cudaGetErrorString(status);
    (void)flag.release();
  }
};

// Frees a 2048_B backing allocation from the fixture's upstream when its owner dies: a fatal
// ASSERT between allocation and test end would otherwise skip the free and leak the GPU
// allocation. Both users allocate exactly 2048_B.
struct backing_free {
  rmm::mr::cuda_memory_resource* upstream;
  void operator()(void* p) const
  {
    if (p) { upstream->deallocate_sync(p, 2048_B, rmm::CUDA_ALLOCATION_ALIGNMENT); }
  }
};

// ---------------------------------------------------------------------------
// Shared fixture with a real upstream
// ---------------------------------------------------------------------------

class FixedSlabTest : public FixedSlabMemoryResourceTest {
 protected:
  rmm::cuda_stream construction_stream{rmm::cuda_stream::flags::non_blocking};
  rmm::mr::cuda_memory_resource upstream{};
};

TEST_F(FixedSlabTest, CarvesAlignedDisjointRanges)
{
  fixed_slab_mr mr{upstream, construction_stream, 1024_B};

  EXPECT_EQ(mr.slab_bytes(), 1024_B);
  EXPECT_EQ(mr.carved_bytes(), 0);

  auto* a = mr.allocate(construction_stream, 1);
  auto* b = mr.allocate(construction_stream, 256_B);
  auto* c = mr.allocate(construction_stream, 7, 64);

  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  ASSERT_NE(c, nullptr);
  EXPECT_TRUE(rmm::test::is_properly_aligned(a));
  EXPECT_TRUE(rmm::test::is_properly_aligned(b));
  EXPECT_TRUE(rmm::is_pointer_aligned(c, 64));
  EXPECT_TRUE(rmm::test::is_device_accessible_memory(a));

  // With a 256-aligned base the starts are 0, 256, 512; no range overlaps.
  auto const base = reinterpret_cast<std::uintptr_t>(a);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(b) - base, 256_B);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(c) - base, 512_B);

  EXPECT_EQ(mr.carved_bytes(), 519);
  EXPECT_EQ(mr.slab_bytes(), 1024_B);

  mr.deallocate(construction_stream, a, 1);
  mr.deallocate(construction_stream, b, 256_B);
  mr.deallocate(construction_stream, c, 7, 64);
  EXPECT_EQ(mr.carved_bytes(), 519);  // no-op deallocation does not restore capacity
}

TEST_F(FixedSlabTest, ExactCapacityRepeatedExhaustionAndNoReuse)
{
  fixed_slab_mr mr{upstream, construction_stream, 512_B};

  auto* a = mr.allocate(construction_stream, 255);
  auto* b = mr.allocate(construction_stream, 256_B);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(mr.carved_bytes(), 512_B);

  mr.deallocate(construction_stream, a, 255);
  mr.deallocate(construction_stream, b, 256_B);
  EXPECT_EQ(mr.carved_bytes(), 512_B);

  // No reuse: after the no-op frees the slab is still exhausted, deterministically.
  EXPECT_THROW((void)mr.allocate(construction_stream, 1), rmm::out_of_memory);
  EXPECT_THROW((void)mr.allocate(construction_stream, 1), rmm::out_of_memory);
  EXPECT_THROW((void)mr.allocate(construction_stream, 1), std::bad_alloc);
  EXPECT_EQ(mr.carved_bytes(), 512_B);

  EXPECT_EQ(mr.allocate(construction_stream, 0), nullptr);
  EXPECT_EQ(mr.allocate_sync(0), nullptr);
  EXPECT_EQ(mr.carved_bytes(), 512_B);
}

TEST_F(FixedSlabTest, FullSlabCarveSucceedsOnce)
{
  fixed_slab_mr mr{upstream, construction_stream, 512_B};

  auto* a = mr.allocate(construction_stream, 512_B);
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(mr.carved_bytes(), 512_B);
  EXPECT_THROW((void)mr.allocate(construction_stream, 1), rmm::out_of_memory);
}

TEST_F(FixedSlabTest, HugeRequestDoesNotConsumeSpace)
{
  fixed_slab_mr mr{upstream, construction_stream, 512_B};

  EXPECT_THROW((void)mr.allocate(construction_stream, std::numeric_limits<std::size_t>::max()),
               rmm::out_of_memory);
  EXPECT_EQ(mr.carved_bytes(), 0);

  // A failed request must leave the cursor untouched.
  auto* a = mr.allocate(construction_stream, 256_B);
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(mr.carved_bytes(), 256_B);
}

TEST_F(FixedSlabTest, ZeroCarveAndInvalidAlignment)
{
  fixed_slab_mr mr{upstream, construction_stream, 512_B};

  EXPECT_EQ(mr.allocate(construction_stream, 0), nullptr);
  EXPECT_EQ(mr.allocate_sync(0), nullptr);
  EXPECT_EQ(mr.carved_bytes(), 0);

  // Null/zero frees never change the cursor.
  mr.deallocate(construction_stream, nullptr, 128);
  mr.deallocate_sync(nullptr, 128);
  EXPECT_EQ(mr.carved_bytes(), 0);

  EXPECT_THROW((void)mr.allocate(construction_stream, 1, 0), rmm::logic_error);
  EXPECT_THROW((void)mr.allocate(construction_stream, 1, 3), rmm::logic_error);
  EXPECT_THROW((void)mr.allocate_sync(1, 768), rmm::logic_error);
  EXPECT_EQ(mr.carved_bytes(), 0);

  mr.deallocate(construction_stream, mr.allocate(construction_stream, 256_B), 0);
  EXPECT_EQ(mr.carved_bytes(), 256_B);
}

TEST_F(FixedSlabTest, SubAllocationAlignmentRequestUsesAlignmentFloor)
{
  fixed_slab_mr mr{upstream, construction_stream, 1024_B};

  // Advance the cursor off the base, then request a sub-floor alignment: the documented
  // alignment floor is `rmm::CUDA_ALLOCATION_ALIGNMENT`, so the slice must come back at
  // that alignment, with floor-derived padding counted in `carved_bytes`.
  auto* a = mr.allocate(construction_stream, 1);
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(mr.carved_bytes(), 1);

  auto* b = mr.allocate(construction_stream, 1, 64);
  ASSERT_NE(b, nullptr);
  EXPECT_TRUE(rmm::is_pointer_aligned(b, rmm::CUDA_ALLOCATION_ALIGNMENT));
  EXPECT_EQ(mr.carved_bytes(), 257);  // 1 + 255 floor padding + 1
}

TEST_F(FixedSlabTest, StreamMismatchThrows)
{
  rmm::cuda_stream other_stream{rmm::cuda_stream::flags::non_blocking};
  fixed_slab_mr mr{upstream, construction_stream, 512_B};

  auto* a = mr.allocate(construction_stream, 256_B);
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(mr.carved_bytes(), 256_B);

  // Single-stream contract: every stream mismatch on a stream-taking call is a caller bug
  // and throws, leaving the cursor untouched (validation precedes the carve arithmetic) —
  // including the zero-byte edge case. allocate_sync takes no stream (nothing to mismatch);
  // deallocate is noexcept and terminates on mismatch (fail-closed, untestable here);
  // deallocate_sync is a no-op.
  EXPECT_THROW((void)mr.allocate(other_stream, 256_B), rmm::logic_error);
  EXPECT_THROW((void)mr.allocate(other_stream, 0), rmm::logic_error);
  EXPECT_THROW((void)mr.allocate(cuda::stream_ref{cudaStreamLegacy}, 256_B), rmm::logic_error);
  EXPECT_EQ(mr.carved_bytes(), 256_B);
}

TEST_F(FixedSlabTest, NullOrZeroDeallocateIgnoredWithoutValidation)
{
  rmm::cuda_stream other_stream{rmm::cuda_stream::flags::non_blocking};
  fixed_slab_mr mr{upstream, construction_stream, 512_B};

  // Documented contract: calls with `ptr == nullptr` or `bytes == 0` are ignored WITHOUT
  // validation, so a wrong stream must NOT reach the fail-closed terminate path here.
  EXPECT_NO_THROW(mr.deallocate(other_stream, nullptr, 0));
  EXPECT_NO_THROW(mr.deallocate(other_stream, nullptr, 256_B));
  EXPECT_NO_THROW(mr.deallocate(other_stream, reinterpret_cast<void*>(0xDEADBEEF), 0));
  EXPECT_NO_THROW(mr.deallocate_sync(nullptr, 0, rmm::CUDA_ALLOCATION_ALIGNMENT));
}

TEST_F(FixedSlabTest, RecordedStreamAccepted)
{
  // Every operation on the recorded stream succeeds; nullptr normalizes to legacy, so a
  // legacy-recorded slab also accepts the null stream handle.
  fixed_slab_mr mr{upstream, construction_stream, 1024_B};

  auto* a = mr.allocate(construction_stream, 256_B);
  ASSERT_NE(a, nullptr);
  EXPECT_TRUE(rmm::test::is_properly_aligned(a));
  mr.deallocate(construction_stream, a, 256_B);
  EXPECT_EQ(mr.carved_bytes(), 256_B);

  // CUDA guarantees only 256-byte alignment, so the padding the 512-carve needs depends on the
  // real base address.
  auto const pad512 = (512 - ((reinterpret_cast<std::uintptr_t>(mr.data()) + 256) & 511)) & 511;

  auto* b = mr.allocate(construction_stream, 256_B, 512);
  ASSERT_NE(b, nullptr);
  EXPECT_TRUE(rmm::is_pointer_aligned(b, 512));
  EXPECT_EQ(mr.carved_bytes(), 512_B + pad512);

  auto* sync_ptr = mr.allocate_sync(1);
  ASSERT_NE(sync_ptr, nullptr);
  EXPECT_EQ(mr.carved_bytes(), 512_B + pad512 + 1);
  mr.deallocate_sync(sync_ptr, 1);
  EXPECT_EQ(mr.carved_bytes(), 512_B + pad512 + 1);
}

TEST_F(FixedSlabTest, SyncAllocationIsHostOnlyCarve)
{
  // allocate_sync performs NO wait: block the recorded stream behind a spinning host
  // callback (with pending construction work ahead of it) and carve synchronously. The
  // carve must return promptly WITHOUT the callback ever running; if a wait were performed
  // the test would hang (or time out) until the flag release.
  // Declared BEFORE the guard so ~mr destructs SECOND: locals unwind in reverse order, so
  // ~unblock sets the flag FIRST and the resource's release() (upstream cudaFree can
  // implicit-synchronize the device) cannot block behind the still-spinning callback.
  fixed_slab_mr mr{upstream, construction_stream, 512_B};
  release_on_exit unblock{construction_stream.value()};
  RMM_CUDA_TRY(
    cudaLaunchHostFunc(construction_stream.value(), spin_until_released, unblock.flag.get()));

  void* ptr = nullptr;
  EXPECT_NO_THROW(ptr = mr.allocate_sync(256_B));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(mr.carved_bytes(), 256_B);
}

TEST_F(FixedSlabTest, UpstreamAllocationErrorPropagates)
{
  mock_resource mock;
  mock_resource_wrapper wrapper{&mock};
  // A null upstream result means the allocation failed; no slab is carved and no free
  // follows.
  EXPECT_CALL(mock, allocate).WillOnce(::testing::Return(nullptr));
  EXPECT_CALL(mock, deallocate).Times(0);

  EXPECT_THROW(((void)fixed_slab_mr{wrapper, cuda::stream_ref{cudaStreamLegacy}, 512_B}),
               rmm::out_of_memory);
}

TEST_F(FixedSlabTest, SharedOwnerOutlivesOriginal)
{
  // An ISOLATED statistics adaptor per assertion group keeps the contract assertions exact:
  // every allocation/free on `shared_stats` belongs to the shared-ownership graph below,
  // and `fresh_stats` starts clean for the independent-slab checks.
  statistics_mr shared_stats{rmm::device_async_resource_ref{upstream}};
  auto const slab_size = 512_B;

  // Exactly ONE upstream allocation for this graph: `copy` and the type-erased `erased`
  // handle must SHARE the impl behind `original` (CCCL shared_resource refcounts the
  // control block), not construct independent slabs.
  std::optional<fixed_slab_mr> original{
    std::in_place, shared_stats, construction_stream, slab_size};
  auto* keep = original->allocate(construction_stream, 256_B);
  ASSERT_NE(keep, nullptr);

  fixed_slab_mr copy                                         = *original;
  cuda::mr::any_resource<cuda::mr::device_accessible> erased = *original;

  // Shared ownership: identical state and identical impl identity (equality is impl
  // identity for this resource).
  EXPECT_EQ(copy.slab_bytes(), slab_size);
  EXPECT_EQ(copy.carved_bytes(), 256_B);
  EXPECT_TRUE(copy == *original);
  EXPECT_TRUE(erased == *original);

  // The original dies here; the shared owners keep the SAME slab alive: the allocation is
  // still outstanding (bytes counter value equals slab_size, never freed) while a
  // surviving owner carves the next offset.
  original.reset();
  EXPECT_EQ(shared_stats.get_bytes_counter().value, slab_size);
  EXPECT_EQ(shared_stats.get_allocations_counter().value, 1);
  auto* next = copy.allocate(construction_stream, 256_B);
  ASSERT_NE(next, nullptr);
  EXPECT_EQ(copy.carved_bytes(), 512_B);

  // The type-erased handle cannot be converted back to the concrete type (CCCL any_resource
  // is one-way), so the shared-impl semantics are already proven by the equality checks
  // above and by `copy`'s post-`reset()` carve.

  // Replace the last owner's two references with throwaway slabs: the original's slab is
  // now fully released. Exact counts for THIS graph: 3 slabs constructed through
  // `shared_stats` (original + the two throwaways), 2 still outstanding.
  copy   = fixed_slab_mr{shared_stats, construction_stream, slab_size};
  erased = fixed_slab_mr{shared_stats, construction_stream, slab_size};
  EXPECT_EQ(shared_stats.get_bytes_counter().value, 2 * slab_size);
  EXPECT_EQ(shared_stats.get_bytes_counter().total, 3 * slab_size);
  EXPECT_EQ(shared_stats.get_allocations_counter().total, 3);

  // A fresh independent slab is a different resource (its own impl, own counters), and its
  // first carve succeeds from the base; its allocations never touch `shared_stats`.
  statistics_mr fresh_stats{rmm::device_async_resource_ref{upstream}};
  fixed_slab_mr other{fresh_stats, construction_stream, slab_size};
  fixed_slab_mr another{fresh_stats, construction_stream, slab_size};
  EXPECT_FALSE(other == another);
  EXPECT_FALSE(other == copy);
  auto* first = other.allocate(construction_stream, 256_B);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(other.carved_bytes(), 256_B);
  EXPECT_EQ(fresh_stats.get_allocations_counter().value, 2);
  EXPECT_EQ(shared_stats.get_allocations_counter().value, 2);
}

TEST_F(FixedSlabTest, DeallocateAfterResourceDestroyed)
{
  // Buffers carved from the slab remain deallocatable AFTER the constructing resource is
  // destroyed: the refcounted impl outlives the concrete wrapper, and a type-erased owner
  // (any_resource) can carve and free through it. The statistics adaptor proves the
  // upstream accounting balances exactly when the LAST owner dies — the slices themselves
  // never touch the upstream.
  statistics_mr stats{rmm::device_async_resource_ref{upstream}};
  std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> erased;
  {
    fixed_slab_mr mr{stats, construction_stream, 768_B};
    erased.emplace(mr);
    auto* a = mr.allocate(construction_stream, 256_B);
    ASSERT_NE(a, nullptr);
    (void)erased->allocate(construction_stream, 128_B, rmm::CUDA_ALLOCATION_ALIGNMENT);
    // Only the slab passes through the adaptor; slices are host-only carves.
    EXPECT_EQ(stats.get_bytes_counter().value, 768_B);
    EXPECT_EQ(stats.get_allocations_counter().value, 1);
  }  // The concrete wrapper dies here; the erased owner keeps the SAME slab alive.

  // Deallocation through the type-erased owner after the wrapper is gone: no-op on the
  // memory, no crash, and still no upstream free (the slab is owned by the erased handle).
  auto* b = erased->allocate(construction_stream, 128_B, rmm::CUDA_ALLOCATION_ALIGNMENT);
  ASSERT_NE(b, nullptr);
  erased->deallocate(construction_stream, b, 128_B, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_EQ(stats.get_bytes_counter().value, 768_B);

  // Last owner dies NOW: the single upstream free fires and the counters zero out.
  erased.reset();
  EXPECT_EQ(stats.get_bytes_counter().value, 0);
  EXPECT_EQ(stats.get_allocations_counter().value, 0);
  EXPECT_EQ(stats.get_bytes_counter().total, 768_B);
  EXPECT_EQ(stats.get_allocations_counter().total, 1);
}

TEST_F(FixedSlabTest, SlabFreedExactlyAfterLastOwnerDies)
{
  // The core lifecycle contract as one linear narrative: the constructing scope dies
  // FIRST, surviving owners keep the slab alive (upstream free count stays 0), per-buffer
  // deallocation is a no-op (count stays 0), and the single upstream free happens EXACTLY
  // when the last owner dies, with the exact (stream, base, slab_bytes, alignment) the
  // construction recorded — never before, never twice. StrictMock makes any premature or
  // duplicate upstream free an immediate test failure.
  ::testing::StrictMock<mock_resource> mock;
  rmm::mr::cuda_memory_resource real{};
  void* base_ptr{nullptr};
  int upstream_frees{0};
  ON_CALL(mock, allocate)
    .WillByDefault(
      ::testing::Invoke([&real](cuda::stream_ref stream, std::size_t bytes, std::size_t alignment) {
        return real.allocate(stream, bytes, alignment);
      }));
  // Strict expectations must be installed BEFORE the resource is constructed (the
  // constructor's allocate hits the mock, and a StrictMock reports unanticipated calls as
  // fatal). Exactly one allocate at construction, one deallocate at last-owner death.
  EXPECT_CALL(mock, allocate).Times(1);
  ON_CALL(mock, deallocate)
    .WillByDefault(::testing::Invoke(
      [&real, &upstream_frees](
        cuda::stream_ref stream, void* ptr, std::size_t bytes, std::size_t alignment) {
        ++upstream_frees;
        real.deallocate(stream, ptr, bytes, alignment);
      }));

  std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> owner;
  void* a{nullptr};
  void* b{nullptr};
  {
    fixed_slab_mr mr{mock_resource_wrapper{&mock}, construction_stream, 768_B};
    base_ptr = mr.data();
    a        = mr.allocate(construction_stream, 256_B);
    b        = mr.allocate(construction_stream, 384_B);
    ASSERT_NE(base_ptr, nullptr);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    owner.emplace(mr);
  }  // Scope death: the owning buffers keep the slab; NO upstream free yet.

  EXPECT_EQ(upstream_frees, 0);
  owner->deallocate(construction_stream, a, 256_B, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_EQ(upstream_frees, 0);  // Buffer A's deallocate is a no-op, not an upstream free.
  owner->deallocate(construction_stream, b, 384_B, rmm::CUDA_ALLOCATION_ALIGNMENT);
  EXPECT_EQ(upstream_frees, 0);  // Same for buffer B.

  // The one and only upstream free, matched exactly: fired by the last owner's death.
  EXPECT_CALL(mock,
              deallocate(cuda::stream_ref{construction_stream.value()},
                         base_ptr,
                         768_B,
                         rmm::CUDA_ALLOCATION_ALIGNMENT))
    .Times(1);
  owner.reset();
  EXPECT_EQ(upstream_frees, 1);
}

TEST_F(FixedSlabTest, DataAccessor)
{
  // data() returns the slab base; the first carve starts at that base (offset 0).
  fixed_slab_mr mr{upstream, construction_stream, 512_B};
  auto* base = mr.data();
  ASSERT_NE(base, nullptr);
  EXPECT_TRUE(rmm::is_pointer_aligned(base, rmm::CUDA_ALLOCATION_ALIGNMENT));
  EXPECT_EQ(mr.slab_bytes(), 512_B);

  auto* first = mr.allocate(construction_stream, 256_B);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(static_cast<std::byte*>(first), static_cast<std::byte*>(base));
  EXPECT_EQ(static_cast<std::byte*>(base) + 256_B, mr.allocate(construction_stream, 256_B));

  // Shared copies of the same slab report the SAME base pointer.
  fixed_slab_mr copy = mr;
  EXPECT_EQ(copy.data(), base);
  EXPECT_TRUE(copy == mr);

  // Independent slabs report DIFFERENT base pointers.
  fixed_slab_mr other{upstream, construction_stream, 512_B};
  EXPECT_NE(other.data(), base);
  EXPECT_FALSE(other == mr);
}

TEST_F(FixedSlabTest, DeviceAccessible)
{
  static_assert(cuda::mr::resource_with<fixed_slab_mr, cuda::mr::device_accessible>);
  static_assert(cuda::mr::synchronous_resource_with<fixed_slab_mr, cuda::mr::device_accessible>);
  static_assert(std::is_copy_constructible_v<fixed_slab_mr>);

  // 768_B slab: the three handles below carve 256 + 256 + 1 = 513 bytes total, so the
  // base-handle carve succeeds against the SHARED slab (all handles alias one impl).
  fixed_slab_mr mr{upstream, construction_stream, 768_B};

  // Type-erased async path with an explicit alignment argument.
  cuda::mr::any_resource<cuda::mr::device_accessible> erased = mr;
  auto* a = erased.allocate(construction_stream, 256_B, rmm::CUDA_ALLOCATION_ALIGNMENT);
  ASSERT_NE(a, nullptr);
  erased.deallocate(construction_stream, a, 256_B, rmm::CUDA_ALLOCATION_ALIGNMENT);

  // RMM ref path.
  rmm::device_async_resource_ref ref = mr;
  auto* b = ref.allocate(construction_stream, 256_B, rmm::CUDA_ALLOCATION_ALIGNMENT);
  ASSERT_NE(b, nullptr);
  ref.deallocate(construction_stream, b, 256_B, rmm::CUDA_ALLOCATION_ALIGNMENT);

  // Direct CCCL base handle exercises the out-of-line impl symbols against shared librmm.
  cuda::mr::shared_resource<rmm::mr::detail::fixed_slab_memory_resource_impl> base_handle = mr;
  auto* c = base_handle.allocate(construction_stream, 1, rmm::CUDA_ALLOCATION_ALIGNMENT);
  ASSERT_NE(c, nullptr);
  base_handle.deallocate(construction_stream, c, 1, rmm::CUDA_ALLOCATION_ALIGNMENT);

  EXPECT_EQ(mr.carved_bytes(), 513);
}

TEST_F(FixedSlabTest, SingleExactUpstreamAllocationAndFree)
{
  // Strict mock delegating to the real resource (real allocation: memory is accessed).
  // Exactly ONE allocate(stream, 768, 256) at construction and ONE
  // deallocate(stream, base, 768, 256) at final destruction; carves, sub-deallocation and
  // exhaustion never touch the upstream.
  ::testing::StrictMock<mock_resource> mock;
  rmm::mr::cuda_memory_resource real{};
  void* base_ptr{nullptr};
  ON_CALL(mock, allocate)
    .WillByDefault(
      ::testing::Invoke([&real](cuda::stream_ref stream, std::size_t bytes, std::size_t alignment) {
        return real.allocate(stream, bytes, alignment);
      }));
  ON_CALL(mock, deallocate)
    .WillByDefault(::testing::Invoke(
      [&real](cuda::stream_ref stream, void* ptr, std::size_t bytes, std::size_t alignment) {
        real.deallocate(stream, ptr, bytes, alignment);
      }));

  EXPECT_CALL(
    mock,
    allocate(cuda::stream_ref{construction_stream.value()}, 768_B, rmm::CUDA_ALLOCATION_ALIGNMENT))
    .Times(1)
    .WillOnce(
      ::testing::Invoke([&real](cuda::stream_ref stream, std::size_t bytes, std::size_t alignment) {
        return real.allocate(stream, bytes, alignment);
      }));

  {
    fixed_slab_mr mr{mock_resource_wrapper{&mock}, construction_stream, 768_B};
    base_ptr = mr.allocate(construction_stream, 256_B);
    ASSERT_NE(base_ptr, nullptr);
    mr.deallocate(construction_stream, base_ptr, 256_B);
    // 512_B still fit exactly, so the exhaustion probe is one byte past the remaining
    // capacity; it throws and consumes nothing.
    EXPECT_THROW((void)mr.allocate(construction_stream, 513), rmm::out_of_memory);
    EXPECT_EQ(mr.carved_bytes(), 256_B);

    // Set only now: any deallocate BEFORE this point (i.e. during the lifetime) is a
    // strict-mock failure; the single free must come from final destruction.
    EXPECT_CALL(mock,
                deallocate(cuda::stream_ref{construction_stream.value()},
                           base_ptr,
                           768_B,
                           rmm::CUDA_ALLOCATION_ALIGNMENT))
      .Times(1);
  }
}

TEST_F(FixedSlabTest, HonorsLargerAlignmentAtMisalignedBase)
{
  // A mock upstream hands out a DETERMINISTIC misaligned base: 256-aligned but NOT
  // 512-aligned (real backing + 256 bytes), so neither test depends on cudaMalloc
  // accidentally returning an odd multiple of 256. No GTEST_SKIP fallback exists.
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  auto* backing = upstream.allocate_sync(2048_B, rmm::CUDA_ALLOCATION_ALIGNMENT);
  ASSERT_NE(backing, nullptr);
  std::unique_ptr<std::byte, backing_free> backing_owner{reinterpret_cast<std::byte*>(backing),
                                                         backing_free{&upstream}};
  auto const backing_address = reinterpret_cast<std::uintptr_t>(backing);
  auto* misaligned           = reinterpret_cast<std::byte*>(
    (backing_address % 512 == 0) ? backing_address + 256 : backing_address);
  ASSERT_TRUE(rmm::is_pointer_aligned(misaligned, 256));
  ASSERT_FALSE(rmm::is_pointer_aligned(misaligned, 512));

  ::testing::StrictMock<mock_resource> mock;
  EXPECT_CALL(mock, allocate(::testing::_, 1024_B, rmm::CUDA_ALLOCATION_ALIGNMENT))
    .WillOnce(::testing::Return(static_cast<void*>(misaligned)));
  // The slab base is handed back verbatim at teardown (strict: no other call).
  EXPECT_CALL(
    mock,
    deallocate(
      ::testing::_, static_cast<void*>(misaligned), 1024_B, rmm::CUDA_ALLOCATION_ALIGNMENT))
    .Times(1);

  {
    fixed_slab_mr mr{mock_resource_wrapper{&mock}, stream, 1024_B};
    // The first 512-aligned carve must start at base+256 (the padding consumes the
    // misalignment), so carved_bytes counts the padding plus the byte.
    auto* a = mr.allocate(stream, 1, 512);
    ASSERT_NE(a, nullptr);
    EXPECT_TRUE(rmm::is_pointer_aligned(a, 512));
    EXPECT_EQ(a, misaligned + 256);
    EXPECT_EQ(mr.carved_bytes(), 257);
  }
}

TEST_F(FixedSlabTest, PaddingDoesNotFit)
{
  // A 256 B slab with a DETERMINISTICALLY misaligned-to-512 base cannot serve 1 byte at
  // 512 alignment: the padding alone is 256 B, which exceeds the whole slab. The request
  // throws and consumes nothing; a plain 256-byte carve still fits. The mock supplies the
  // misaligned base, so no GTEST_SKIP fallback or allocator luck is involved.
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  auto* backing = upstream.allocate_sync(2048_B, rmm::CUDA_ALLOCATION_ALIGNMENT);
  ASSERT_NE(backing, nullptr);
  std::unique_ptr<std::byte, backing_free> backing_owner{reinterpret_cast<std::byte*>(backing),
                                                         backing_free{&upstream}};
  auto const backing_address = reinterpret_cast<std::uintptr_t>(backing);
  auto* misaligned           = reinterpret_cast<std::byte*>(
    (backing_address % 512 == 0) ? backing_address + 256 : backing_address);
  ASSERT_TRUE(rmm::is_pointer_aligned(misaligned, 256));
  ASSERT_FALSE(rmm::is_pointer_aligned(misaligned, 512));

  ::testing::StrictMock<mock_resource> mock;
  EXPECT_CALL(mock, allocate(::testing::_, 256_B, rmm::CUDA_ALLOCATION_ALIGNMENT))
    .WillOnce(::testing::Return(static_cast<void*>(misaligned)));
  EXPECT_CALL(
    mock,
    deallocate(::testing::_, static_cast<void*>(misaligned), 256_B, rmm::CUDA_ALLOCATION_ALIGNMENT))
    .Times(1);

  {
    fixed_slab_mr mr{mock_resource_wrapper{&mock}, stream, 256_B};
    EXPECT_THROW((void)mr.allocate(stream, 1, 512), rmm::out_of_memory);
    EXPECT_EQ(mr.carved_bytes(), 0);

    auto* a = mr.allocate(stream, 256_B);
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(a, misaligned);
    EXPECT_EQ(mr.carved_bytes(), 256_B);
  }
}

TEST_F(FixedSlabTest, ExplicitDefaultStreams)
{
  // The null stream is a documented alias of the legacy stream, so both keys land on the
  // same normalized stream and the slices are disjoint on that single stream.
  fixed_slab_mr mr{upstream, cuda::stream_ref{cudaStream_t{cudaStreamLegacy}}, 768_B};

  auto* legacy = mr.allocate(cuda::stream_ref{cudaStreamLegacy}, 256_B);
  auto* null   = mr.allocate(cuda::stream_ref{cudaStream_t{nullptr}}, 256_B);
  ASSERT_NE(legacy, nullptr);
  ASSERT_NE(null, nullptr);
  EXPECT_NE(legacy, null);
  EXPECT_EQ(mr.carved_bytes(), 512_B);

  // The sync pair also works against the same resource; the remaining 256 B carve
  // synchronously succeeds and is host-ready on return. The slab is now exactly full.
  auto* sync_ptr = mr.allocate_sync(256_B, 256_B);
  ASSERT_NE(sync_ptr, nullptr);
  EXPECT_TRUE(rmm::test::is_properly_aligned(sync_ptr));
  EXPECT_EQ(mr.carved_bytes(), 768_B);
  // Exhaustion boundary: zero capacity remain, so even a 1-byte carve throws OOM and
  // consumes nothing.
  EXPECT_THROW((void)mr.allocate(cuda::stream_ref{cudaStreamLegacy}, 1), rmm::out_of_memory);
  EXPECT_THROW((void)mr.allocate_sync(1), rmm::out_of_memory);
  EXPECT_EQ(mr.carved_bytes(), 768_B);

  mr.deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, legacy, 256_B);
  mr.deallocate(cuda::stream_ref{cudaStreamLegacy}, null, 256_B);
  mr.deallocate_sync(sync_ptr, 256_B);
  EXPECT_EQ(mr.carved_bytes(), 768_B);
}

TEST_F(FixedSlabTest, PerThreadStreamAsRecordedStream)
{
  // Explicit cudaStreamPerThread can BE the recorded stream: every call then uses it, and
  // a different stream still throws. This target is registered through ConfigureTestInternal,
  // so no PTDS-compiled binary variant exists; this in-binary coverage exercises the PTDS
  // semantic directly.
  fixed_slab_mr mr{upstream, cuda::stream_ref{cudaStreamPerThread}, 512_B};

  auto* a = mr.allocate(cuda::stream_ref{cudaStreamPerThread}, 256_B);
  ASSERT_NE(a, nullptr);
  EXPECT_TRUE(rmm::test::is_properly_aligned(a));
  mr.deallocate(cuda::stream_ref{cudaStreamPerThread}, a, 256_B);
  EXPECT_EQ(mr.carved_bytes(), 256_B);

  rmm::cuda_stream other_stream{rmm::cuda_stream::flags::non_blocking};
  EXPECT_THROW((void)mr.allocate(other_stream, 256_B), rmm::logic_error);
  EXPECT_EQ(mr.carved_bytes(), 256_B);
}

}  // namespace
}  // namespace rmm::test
