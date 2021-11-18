/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <rmm/detail/aligned.hpp>
#include <rmm/detail/cuda_util.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include "../../byte_literals.hpp"

#include <gmock/gmock-actions.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace rmm::test {
namespace {

class mock_memory_resource {
 public:
  MOCK_METHOD(void*, allocate, (std::size_t));
  MOCK_METHOD(void, deallocate, (void*, std::size_t));
};

using memory_span  = rmm::mr::detail::arena::memory_span;
using block        = rmm::mr::detail::arena::block;
using superblock   = rmm::mr::detail::arena::superblock;
using global_arena = rmm::mr::detail::arena::global_arena<mock_memory_resource>;
using arena        = rmm::mr::detail::arena::arena<mock_memory_resource>;
using arena_mr     = rmm::mr::arena_memory_resource<rmm::mr::device_memory_resource>;
using ::testing::Return;

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
auto const fake_address = reinterpret_cast<void*>(1_KiB);
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
auto const fake_address2 = reinterpret_cast<void*>(2_KiB);
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
auto const fake_address3 = reinterpret_cast<void*>(4_MiB);
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
auto const fake_address4 = reinterpret_cast<void*>(8_MiB);

/**
 * Test memory_span.
 */

TEST(ArenaTest, MemorySpan)  // NOLINT
{
  memory_span const ms{};
  EXPECT_FALSE(ms.is_valid());
  memory_span const ms2{fake_address, 256};
  EXPECT_TRUE(ms2.is_valid());
}

/**
 * Test block.
 */

TEST(ArenaTest, BlockFits)  // NOLINT
{
  block const b{fake_address, 1_KiB};
  EXPECT_TRUE(b.fits(1_KiB));
  EXPECT_FALSE(b.fits(1_KiB + 1));
}

TEST(ArenaTest, BlockIsContiguousBefore)  // NOLINT
{
  block const b{fake_address, 1_KiB};
  block const b2{fake_address2, 256};
  EXPECT_TRUE(b.is_contiguous_before(b2));
  block const b3{fake_address, 512};
  block const b4{fake_address2, 1_KiB};
  EXPECT_FALSE(b3.is_contiguous_before(b4));
}

TEST(ArenaTest, BlockSplit)  // NOLINT
{
  block const b{fake_address, 2_KiB};
  auto const [head, tail] = b.split(1_KiB);
  EXPECT_EQ(head.pointer(), fake_address);
  EXPECT_EQ(head.size(), 1_KiB);
  EXPECT_EQ(tail.pointer(), fake_address2);
  EXPECT_EQ(tail.size(), 1_KiB);
}

TEST(ArenaTest, BlockMerge)  // NOLINT
{
  block const b{fake_address, 1_KiB};
  block const b2{fake_address2, 1_KiB};
  auto const merged = b.merge(b2);
  EXPECT_EQ(merged.pointer(), fake_address);
  EXPECT_EQ(merged.size(), 2_KiB);
}

/**
 * Test superblock.
 */

TEST(ArenaTest, SuperblockEmpty)  // NOLINT
{
  superblock sb{fake_address3, 4_MiB};
  EXPECT_TRUE(sb.empty());
  sb.first_fit(256);
  EXPECT_FALSE(sb.empty());
}

TEST(ArenaTest, SuperblockContains)  // NOLINT
{
  superblock const sb{fake_address3, 4_MiB};
  block const b{fake_address, 2_KiB};
  EXPECT_FALSE(sb.contains(b));
  block const b2{fake_address3, 1_KiB};
  EXPECT_TRUE(sb.contains(b2));
  block const b3{fake_address3, 4_MiB + 1};
  EXPECT_FALSE(sb.contains(b3));
  block const b4{fake_address3, 4_MiB};
  EXPECT_TRUE(sb.contains(b4));
  block const b5{fake_address4, 256};
  EXPECT_FALSE(sb.contains(b5));
}

TEST(ArenaTest, SuperblockFits)  // NOLINT
{
  superblock sb{fake_address3, 4_MiB};
  EXPECT_TRUE(sb.fits(4_MiB));
  EXPECT_FALSE(sb.fits(4_MiB + 1));

  auto const b = sb.first_fit(1_MiB);
  sb.first_fit(1_MiB);
  sb.coalesce(b);
  EXPECT_TRUE(sb.fits(2_MiB));
  EXPECT_FALSE(sb.fits(2_MiB + 1));
}

TEST(ArenaTest, SuperblockIsContiguousBefore)  // NOLINT
{
  superblock sb{fake_address3, 4_MiB};
  superblock sb2{fake_address4, 4_MiB};
  EXPECT_TRUE(sb.is_contiguous_before(sb2));

  auto const b = sb.first_fit(256);
  EXPECT_FALSE(sb.is_contiguous_before(sb2));
  sb.coalesce(b);
  EXPECT_TRUE(sb.is_contiguous_before(sb2));

  auto const b2 = sb2.first_fit(1_KiB);
  EXPECT_FALSE(sb.is_contiguous_before(sb2));
  sb2.coalesce(b2);
  EXPECT_TRUE(sb.is_contiguous_before(sb2));
}

TEST(ArenaTest, SuperblockSplit)  // NOLINT
{
  superblock sb{fake_address3, 8_MiB};
  auto const [head, tail] = sb.split(4_MiB);
  EXPECT_EQ(head.pointer(), fake_address3);
  EXPECT_EQ(head.size(), 4_MiB);
  EXPECT_TRUE(head.empty());
  EXPECT_EQ(tail.pointer(), fake_address4);
  EXPECT_EQ(tail.size(), 4_MiB);
  EXPECT_TRUE(tail.empty());
}

TEST(ArenaTest, SuperblockMerge)  // NOLINT
{
  superblock sb{fake_address3, 4_MiB};
  superblock sb2{fake_address4, 4_MiB};
  auto const merged = sb.merge(sb2);
  EXPECT_EQ(merged.pointer(), fake_address3);
  EXPECT_EQ(merged.size(), 8_MiB);
  EXPECT_TRUE(merged.empty());
}

TEST(ArenaTest, SuperblockFirstFit)  // NOLINT
{
  superblock sb{fake_address3, 4_MiB};
  auto const b = sb.first_fit(1_KiB);
  EXPECT_EQ(b.pointer(), fake_address3);
  EXPECT_EQ(b.size(), 1_KiB);
  auto const b2 = sb.first_fit(2_KiB);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  EXPECT_EQ(b2.pointer(), static_cast<char*>(fake_address3) + 1_KiB);
  EXPECT_EQ(b2.size(), 2_KiB);
  sb.coalesce(b);
  auto const b3 = sb.first_fit(512);
  EXPECT_EQ(b3.pointer(), fake_address3);
  EXPECT_EQ(b3.size(), 512);
}

TEST(ArenaTest, SuperblockCoalesceAfterFull)  // NOLINT
{
  superblock sb{fake_address3, 4_MiB};
  auto const b = sb.first_fit(2_MiB);
  sb.first_fit(2_MiB);
  sb.coalesce(b);
  EXPECT_TRUE(sb.first_fit(2_MiB).is_valid());
}

TEST(ArenaTest, SuperblockCoalesceMergeNext)  // NOLINT
{
  superblock sb{fake_address3, 4_MiB};
  auto const b = sb.first_fit(2_MiB);
  sb.coalesce(b);
  EXPECT_TRUE(sb.first_fit(4_MiB).is_valid());
}

TEST(ArenaTest, SuperblockCoalesceMergePrevious)  // NOLINT
{
  superblock sb{fake_address3, 4_MiB};
  auto const b  = sb.first_fit(1_KiB);
  auto const b2 = sb.first_fit(1_KiB);
  sb.first_fit(1_KiB);
  sb.coalesce(b);
  sb.coalesce(b2);
  auto const b3 = sb.first_fit(2_KiB);
  EXPECT_EQ(b3.pointer(), fake_address3);
}

TEST(ArenaTest, SuperblockCoalesceMergePreviousAndNext)  // NOLINT
{
  superblock sb{fake_address3, 4_MiB};
  auto const b  = sb.first_fit(1_KiB);
  auto const b2 = sb.first_fit(1_KiB);
  sb.coalesce(b);
  sb.coalesce(b2);
  EXPECT_TRUE(sb.first_fit(4_MiB).is_valid());
}

TEST(ArenaTest, SuperblockMaxFree)  // NOLINT
{
  superblock sb{fake_address3, 4_MiB};
  sb.first_fit(2_MiB);
  EXPECT_EQ(sb.max_free(), 2_MiB);
}

/**
 * Test global_arena.
 */

TEST(ArenaTest, GlobalArenaNullUpstream)  // NOLINT
{
  auto construct_nullptr = []() { global_arena ga{nullptr, std::nullopt}; };
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);  // NOLINT(cppcoreguidelines-avoid-goto)
}

TEST(ArenaTest, GlobalArenaAcquire)  // NOLINT
{
  mock_memory_resource mock;
  EXPECT_CALL(mock, allocate(8_MiB)).WillOnce(Return(fake_address3));
  EXPECT_CALL(mock, deallocate(fake_address3, 8_MiB));

  global_arena ga{&mock, 8_MiB};

  auto const sb = ga.acquire(256);
  EXPECT_EQ(sb.pointer(), fake_address3);
  EXPECT_EQ(sb.size(), 4_MiB);
  EXPECT_TRUE(sb.empty());

  auto const sb2 = ga.acquire(1_KiB);
  EXPECT_EQ(sb2.pointer(), fake_address4);
  EXPECT_EQ(sb2.size(), 4_MiB);
  EXPECT_TRUE(sb2.empty());

  EXPECT_FALSE(ga.acquire(512).is_valid());
}

TEST(ArenaTest, GlobalArenaReleaseMergeNext)  // NOLINT
{
  mock_memory_resource mock;
  EXPECT_CALL(mock, allocate(8_MiB)).WillOnce(Return(fake_address3));
  EXPECT_CALL(mock, deallocate(fake_address3, 8_MiB));

  global_arena ga{&mock, 8_MiB};

  auto sb = ga.acquire(256);
  ga.release(std::move(sb), {});
  auto* p = ga.allocate(8_MiB);
  EXPECT_EQ(p, fake_address3);
}

TEST(ArenaTest, GlobalArenaReleaseMergePrevious)  // NOLINT
{
  mock_memory_resource mock;
  EXPECT_CALL(mock, allocate(16_MiB)).WillOnce(Return(fake_address3));
  EXPECT_CALL(mock, deallocate(fake_address3, 16_MiB));

  global_arena ga{&mock, 16_MiB};

  auto sb  = ga.acquire(256);
  auto sb2 = ga.acquire(1_KiB);
  ga.acquire(512);
  ga.release(std::move(sb), {});
  ga.release(std::move(sb2), {});
  auto* p = ga.allocate(8_MiB);
  EXPECT_EQ(p, fake_address3);
}

TEST(ArenaTest, GlobalArenaReleaseMergePreviousAndNext)  // NOLINT
{
  mock_memory_resource mock;
  EXPECT_CALL(mock, allocate(16_MiB)).WillOnce(Return(fake_address3));
  EXPECT_CALL(mock, deallocate(fake_address3, 16_MiB));

  global_arena ga{&mock, 16_MiB};

  auto sb  = ga.acquire(256);
  auto sb2 = ga.acquire(1_KiB);
  auto sb3 = ga.acquire(512);
  ga.release(std::move(sb), {});
  ga.release(std::move(sb3), {});
  ga.release(std::move(sb2), {});
  auto* p = ga.allocate(16_MiB);
  EXPECT_EQ(p, fake_address3);
}

TEST(ArenaTest, GlobalArenaReleaseMultiple)  // NOLINT
{
  mock_memory_resource mock;
  EXPECT_CALL(mock, allocate(16_MiB)).WillOnce(Return(fake_address3));
  EXPECT_CALL(mock, deallocate(fake_address3, 16_MiB));

  global_arena ga{&mock, 16_MiB};

  std::set<superblock> superblocks{};
  auto sb = ga.acquire(256);
  superblocks.insert(std::move(sb));
  auto sb2 = ga.acquire(1_KiB);
  superblocks.insert(std::move(sb2));
  auto sb3 = ga.acquire(512);
  superblocks.insert(std::move(sb3));
  ga.release(superblocks);
  auto* p = ga.allocate(16_MiB);
  EXPECT_EQ(p, fake_address3);
}

TEST(ArenaTest, GlobalArenaAllocate)  // NOLINT
{
  mock_memory_resource mock;
  EXPECT_CALL(mock, allocate(8_MiB)).WillOnce(Return(fake_address3));
  EXPECT_CALL(mock, deallocate(fake_address3, 8_MiB));

  global_arena ga{&mock, 8_MiB};

  auto* ptr = ga.allocate(4_MiB);
  EXPECT_EQ(ptr, fake_address3);
  auto* ptr2 = ga.allocate(4_MiB);
  EXPECT_EQ(ptr2, fake_address4);
}

TEST(ArenaTest, GlobalArenaDeallocate)  // NOLINT
{
  mock_memory_resource mock;
  EXPECT_CALL(mock, allocate(8_MiB)).WillOnce(Return(fake_address3));
  EXPECT_CALL(mock, deallocate(fake_address3, 8_MiB));

  global_arena ga{&mock, 8_MiB};

  auto* ptr = ga.allocate(4_MiB);
  EXPECT_EQ(ptr, fake_address3);
  ga.deallocate(ptr, 4_MiB, {});
  ptr = ga.allocate(4_MiB);
  EXPECT_EQ(ptr, fake_address3);
}

TEST(ArenaTest, GlobalArenaDeallocateFromOtherArena)  // NOLINT
{
  mock_memory_resource mock;
  EXPECT_CALL(mock, allocate(8_MiB)).WillOnce(Return(fake_address3));
  EXPECT_CALL(mock, deallocate(fake_address3, 8_MiB));

  global_arena ga{&mock, 8_MiB};

  auto sb      = ga.acquire(512);
  auto const b = sb.first_fit(512);
  ga.release(std::move(sb), {});
  ga.deallocate_from_other_arena(b.pointer(), b.size());
  EXPECT_EQ(ga.allocate(8_MiB), fake_address3);
}

/**
 * Test arena.
 */

TEST(ArenaTest, ArenaAllocate)  // NOLINT
{
  mock_memory_resource mock;
  EXPECT_CALL(mock, allocate(8_MiB)).WillOnce(Return(fake_address3));
  EXPECT_CALL(mock, deallocate(fake_address3, 8_MiB));
  global_arena ga{&mock, 8_MiB};
  arena a{ga};

  EXPECT_EQ(a.allocate(4_MiB), fake_address3);
  EXPECT_EQ(a.allocate(256), fake_address4);
}

TEST(ArenaTest, ArenaDeallocate)  // NOLINT
{
  mock_memory_resource mock;
  EXPECT_CALL(mock, allocate(8_MiB)).WillOnce(Return(fake_address3));
  EXPECT_CALL(mock, deallocate(fake_address3, 8_MiB));
  global_arena ga{&mock, 8_MiB};
  arena a{ga};

  auto* ptr = a.allocate(4_MiB);
  a.deallocate(ptr, 4_MiB, {});
  auto* ptr2 = a.allocate(256);
  a.deallocate(ptr2, 256, {});
  EXPECT_EQ(a.allocate(8_MiB), fake_address3);
}

TEST(ArenaTest, ArenaDeallocateMergePrevious)  // NOLINT
{
  mock_memory_resource mock;
  EXPECT_CALL(mock, allocate(8_MiB)).WillOnce(Return(fake_address3));
  EXPECT_CALL(mock, deallocate(fake_address3, 8_MiB));
  global_arena ga{&mock, 8_MiB};
  arena a{ga};

  auto* ptr  = a.allocate(256);
  auto* ptr2 = a.allocate(256);
  a.allocate(256);
  a.deallocate(ptr, 256, {});
  a.deallocate(ptr2, 256, {});
  EXPECT_EQ(a.allocate(512), fake_address3);
}

TEST(ArenaTest, ArenaDeallocateMergeNext)  // NOLINT
{
  mock_memory_resource mock;
  EXPECT_CALL(mock, allocate(8_MiB)).WillOnce(Return(fake_address3));
  EXPECT_CALL(mock, deallocate(fake_address3, 8_MiB));
  global_arena ga{&mock, 8_MiB};
  arena a{ga};

  auto* ptr  = a.allocate(256);
  auto* ptr2 = a.allocate(256);
  a.allocate(256);
  a.deallocate(ptr2, 256, {});
  a.deallocate(ptr, 256, {});
  EXPECT_EQ(a.allocate(512), fake_address3);
}

TEST(ArenaTest, ArenaDeallocateMergePreviousAndNext)  // NOLINT
{
  mock_memory_resource mock;
  EXPECT_CALL(mock, allocate(8_MiB)).WillOnce(Return(fake_address3));
  EXPECT_CALL(mock, deallocate(fake_address3, 8_MiB));
  global_arena ga{&mock, 8_MiB};
  arena a{ga};

  auto* ptr  = a.allocate(256);
  auto* ptr2 = a.allocate(256);
  a.deallocate(ptr, 256, {});
  a.deallocate(ptr2, 256, {});
  EXPECT_EQ(a.allocate(2_KiB), fake_address3);
}

/**
 * Test arena_memory_resource.
 */

TEST(ArenaTest, NullUpstream)  // NOLINT
{
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto)
  EXPECT_THROW([]() { arena_mr mr{nullptr}; }(), rmm::logic_error);
}

TEST(ArenaTest, AllocateNinetyPercent)  // NOLINT
{
  EXPECT_NO_THROW([]() {  // NOLINT(cppcoreguidelines-avoid-goto)
    auto const free = rmm::detail::available_device_memory().first;
    auto const ninety_percent =
      rmm::detail::align_up(static_cast<std::size_t>(static_cast<double>(free) * 0.9),
                            rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
    arena_mr mr(rmm::mr::get_current_device_resource(), ninety_percent);
  }());
}

TEST(ArenaTest, SmallMediumLarge)  // NOLINT
{
  EXPECT_NO_THROW([]() {  // NOLINT(cppcoreguidelines-avoid-goto)
    arena_mr mr(rmm::mr::get_current_device_resource());
    auto* small     = mr.allocate(256);
    auto* medium    = mr.allocate(64_MiB);
    auto const free = rmm::detail::available_device_memory().first;
    auto* large     = mr.allocate(free / 3);
    mr.deallocate(small, 256);
    mr.deallocate(medium, 64_MiB);
    mr.deallocate(large, free / 3);
  }());
}

}  // namespace
}  // namespace rmm::test
