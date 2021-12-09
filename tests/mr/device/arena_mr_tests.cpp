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

#include <rmm/cuda_stream.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/cuda_util.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include "../../byte_literals.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sys/stat.h>
#include <memory>

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

// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
auto const fake_address  = reinterpret_cast<void*>(1_KiB);
auto const fake_address2 = reinterpret_cast<void*>(2_KiB);
auto const fake_address3 = reinterpret_cast<void*>(superblock::minimum_size);
auto const fake_address4 = reinterpret_cast<void*>(superblock::minimum_size * 2);
// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)

class ArenaTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    EXPECT_CALL(mock_, allocate(arena_size_)).WillOnce(Return(fake_address3));
    EXPECT_CALL(mock_, deallocate(fake_address3, arena_size_));
    global_arena_ = std::make_unique<global_arena>(&mock_, arena_size_);
    arena_        = std::make_unique<arena>(*global_arena_);
  }

  // NOLINTBEGIN(cppcoreguidelines-non-private-member-variables-in-classes)
  std::size_t arena_size_{superblock::minimum_size * 4};
  mock_memory_resource mock_{};
  std::unique_ptr<global_arena> global_arena_{};
  std::unique_ptr<arena> arena_{};
  // NOLINTEND(cppcoreguidelines-non-private-member-variables-in-classes)
};

/**
 * Test align_to_size_class.
 */
TEST_F(ArenaTest, AlignToSizeClass)  // NOLINT
{
  using rmm::mr::detail::arena::align_to_size_class;
  EXPECT_EQ(align_to_size_class(8), 256);
  EXPECT_EQ(align_to_size_class(256), 256);
  EXPECT_EQ(align_to_size_class(264), 512);
  EXPECT_EQ(align_to_size_class(512), 512);
  EXPECT_EQ(align_to_size_class(17_KiB), 20_KiB);
  EXPECT_EQ(align_to_size_class(13_MiB), 14_MiB);
  EXPECT_EQ(align_to_size_class(2500_MiB), 2560_MiB);
  EXPECT_EQ(align_to_size_class(128_GiB), 128_GiB);
  EXPECT_EQ(align_to_size_class(1_PiB), std::numeric_limits<std::size_t>::max());
}

/**
 * Test memory_span.
 */

TEST_F(ArenaTest, MemorySpan)  // NOLINT
{
  memory_span const mem_span{};
  EXPECT_FALSE(mem_span.is_valid());
  memory_span const ms2{fake_address, 256};
  EXPECT_TRUE(ms2.is_valid());
}

/**
 * Test block.
 */

TEST_F(ArenaTest, BlockFits)  // NOLINT
{
  block const blk{fake_address, 1_KiB};
  EXPECT_TRUE(blk.fits(1_KiB));
  EXPECT_FALSE(blk.fits(1_KiB + 1));
}

TEST_F(ArenaTest, BlockIsContiguousBefore)  // NOLINT
{
  block const blk{fake_address, 1_KiB};
  block const blk2{fake_address2, 256};
  EXPECT_TRUE(blk.is_contiguous_before(blk2));
  block const blk3{fake_address, 512};
  block const blk4{fake_address2, 1_KiB};
  EXPECT_FALSE(blk3.is_contiguous_before(blk4));
}

TEST_F(ArenaTest, BlockSplit)  // NOLINT
{
  block const blk{fake_address, 2_KiB};
  auto const [head, tail] = blk.split(1_KiB);
  EXPECT_EQ(head.pointer(), fake_address);
  EXPECT_EQ(head.size(), 1_KiB);
  EXPECT_EQ(tail.pointer(), fake_address2);
  EXPECT_EQ(tail.size(), 1_KiB);
}

TEST_F(ArenaTest, BlockMerge)  // NOLINT
{
  block const blk{fake_address, 1_KiB};
  block const blk2{fake_address2, 1_KiB};
  auto const merged = blk.merge(blk2);
  EXPECT_EQ(merged.pointer(), fake_address);
  EXPECT_EQ(merged.size(), 2_KiB);
}

/**
 * Test superblock.
 */

TEST_F(ArenaTest, SuperblockEmpty)  // NOLINT
{
  superblock sblk{fake_address3, superblock::minimum_size};
  EXPECT_TRUE(sblk.empty());
  sblk.first_fit(256);
  EXPECT_FALSE(sblk.empty());
}

TEST_F(ArenaTest, SuperblockContains)  // NOLINT
{
  superblock const sblk{fake_address3, superblock::minimum_size};
  block const blk{fake_address, 2_KiB};
  EXPECT_FALSE(sblk.contains(blk));
  block const blk2{fake_address3, 1_KiB};
  EXPECT_TRUE(sblk.contains(blk2));
  block const blk3{fake_address3, superblock::minimum_size + 1};
  EXPECT_FALSE(sblk.contains(blk3));
  block const blk4{fake_address3, superblock::minimum_size};
  EXPECT_TRUE(sblk.contains(blk4));
  block const blk5{fake_address4, 256};
  EXPECT_FALSE(sblk.contains(blk5));
}

TEST_F(ArenaTest, SuperblockFits)  // NOLINT
{
  superblock sblk{fake_address3, superblock::minimum_size};
  EXPECT_TRUE(sblk.fits(superblock::minimum_size));
  EXPECT_FALSE(sblk.fits(superblock::minimum_size + 1));

  auto const blk = sblk.first_fit(superblock::minimum_size / 4);
  sblk.first_fit(superblock::minimum_size / 4);
  sblk.coalesce(blk);
  EXPECT_TRUE(sblk.fits(superblock::minimum_size / 2));
  EXPECT_FALSE(sblk.fits(superblock::minimum_size / 2 + 1));
}

TEST_F(ArenaTest, SuperblockIsContiguousBefore)  // NOLINT
{
  superblock sblk{fake_address3, superblock::minimum_size};
  superblock sb2{fake_address4, superblock::minimum_size};
  EXPECT_TRUE(sblk.is_contiguous_before(sb2));

  auto const blk = sblk.first_fit(256);
  EXPECT_FALSE(sblk.is_contiguous_before(sb2));
  sblk.coalesce(blk);
  EXPECT_TRUE(sblk.is_contiguous_before(sb2));

  auto const blk2 = sb2.first_fit(1_KiB);
  EXPECT_FALSE(sblk.is_contiguous_before(sb2));
  sb2.coalesce(blk2);
  EXPECT_TRUE(sblk.is_contiguous_before(sb2));
}

TEST_F(ArenaTest, SuperblockSplit)  // NOLINT
{
  superblock sblk{fake_address3, superblock::minimum_size * 2};
  auto const [head, tail] = sblk.split(superblock::minimum_size);
  EXPECT_EQ(head.pointer(), fake_address3);
  EXPECT_EQ(head.size(), superblock::minimum_size);
  EXPECT_TRUE(head.empty());
  EXPECT_EQ(tail.pointer(), fake_address4);
  EXPECT_EQ(tail.size(), superblock::minimum_size);
  EXPECT_TRUE(tail.empty());
}

TEST_F(ArenaTest, SuperblockMerge)  // NOLINT
{
  superblock sblk{fake_address3, superblock::minimum_size};
  superblock sb2{fake_address4, superblock::minimum_size};
  auto const merged = sblk.merge(sb2);
  EXPECT_EQ(merged.pointer(), fake_address3);
  EXPECT_EQ(merged.size(), superblock::minimum_size * 2);
  EXPECT_TRUE(merged.empty());
}

TEST_F(ArenaTest, SuperblockFirstFit)  // NOLINT
{
  superblock sblk{fake_address3, superblock::minimum_size};
  auto const blk = sblk.first_fit(1_KiB);
  EXPECT_EQ(blk.pointer(), fake_address3);
  EXPECT_EQ(blk.size(), 1_KiB);
  auto const blk2 = sblk.first_fit(2_KiB);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  EXPECT_EQ(blk2.pointer(), static_cast<char*>(fake_address3) + 1_KiB);
  EXPECT_EQ(blk2.size(), 2_KiB);
  sblk.coalesce(blk);
  auto const blk3 = sblk.first_fit(512);
  EXPECT_EQ(blk3.pointer(), fake_address3);
  EXPECT_EQ(blk3.size(), 512);
}

TEST_F(ArenaTest, SuperblockCoalesceAfterFull)  // NOLINT
{
  superblock sblk{fake_address3, superblock::minimum_size};
  auto const blk = sblk.first_fit(superblock::minimum_size / 2);
  sblk.first_fit(superblock::minimum_size / 2);
  sblk.coalesce(blk);
  EXPECT_TRUE(sblk.first_fit(superblock::minimum_size / 2).is_valid());
}

TEST_F(ArenaTest, SuperblockCoalesceMergeNext)  // NOLINT
{
  superblock sblk{fake_address3, superblock::minimum_size};
  auto const blk = sblk.first_fit(superblock::minimum_size / 2);
  sblk.coalesce(blk);
  EXPECT_TRUE(sblk.first_fit(superblock::minimum_size).is_valid());
}

TEST_F(ArenaTest, SuperblockCoalesceMergePrevious)  // NOLINT
{
  superblock sblk{fake_address3, superblock::minimum_size};
  auto const blk  = sblk.first_fit(1_KiB);
  auto const blk2 = sblk.first_fit(1_KiB);
  sblk.first_fit(1_KiB);
  sblk.coalesce(blk);
  sblk.coalesce(blk2);
  auto const blk3 = sblk.first_fit(2_KiB);
  EXPECT_EQ(blk3.pointer(), fake_address3);
}

TEST_F(ArenaTest, SuperblockCoalesceMergePreviousAndNext)  // NOLINT
{
  superblock sblk{fake_address3, superblock::minimum_size};
  auto const blk  = sblk.first_fit(1_KiB);
  auto const blk2 = sblk.first_fit(1_KiB);
  sblk.coalesce(blk);
  sblk.coalesce(blk2);
  EXPECT_TRUE(sblk.first_fit(superblock::minimum_size).is_valid());
}

TEST_F(ArenaTest, SuperblockMaxFreeSize)  // NOLINT
{
  superblock sblk{fake_address3, superblock::minimum_size};
  sblk.first_fit(superblock::minimum_size / 2);
  EXPECT_EQ(sblk.max_free_size(), superblock::minimum_size / 2);
}

TEST_F(ArenaTest, SuperblockMaxFreeSizeWhenFull)  // NOLINT
{
  superblock sblk{fake_address3, superblock::minimum_size};
  sblk.first_fit(superblock::minimum_size);
  EXPECT_EQ(sblk.max_free_size(), 0);
}

/**
 * Test global_arena.
 */

TEST_F(ArenaTest, GlobalArenaNullUpstream)  // NOLINT
{
  auto construct_nullptr = []() { global_arena global{nullptr, std::nullopt}; };
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);  // NOLINT(cppcoreguidelines-avoid-goto)
}

TEST_F(ArenaTest, GlobalArenaAcquire)  // NOLINT
{
  auto const sblk = global_arena_->acquire(256);
  EXPECT_EQ(sblk.pointer(), fake_address3);
  EXPECT_EQ(sblk.size(), superblock::minimum_size);
  EXPECT_TRUE(sblk.empty());

  auto const sb2 = global_arena_->acquire(1_KiB);
  EXPECT_EQ(sb2.pointer(), fake_address4);
  EXPECT_EQ(sb2.size(), superblock::minimum_size);
  EXPECT_TRUE(sb2.empty());

  global_arena_->acquire(512);
  global_arena_->acquire(512);
  EXPECT_FALSE(global_arena_->acquire(512).is_valid());
}

TEST_F(ArenaTest, GlobalArenaReleaseMergeNext)  // NOLINT
{
  auto sblk = global_arena_->acquire(256);
  global_arena_->release(std::move(sblk));
  auto* ptr = global_arena_->allocate(arena_size_);
  EXPECT_EQ(ptr, fake_address3);
}

TEST_F(ArenaTest, GlobalArenaReleaseMergePrevious)  // NOLINT
{
  auto sblk = global_arena_->acquire(256);
  auto sb2  = global_arena_->acquire(1_KiB);
  global_arena_->acquire(512);
  global_arena_->release(std::move(sblk));
  global_arena_->release(std::move(sb2));
  auto* ptr = global_arena_->allocate(superblock::minimum_size * 2);
  EXPECT_EQ(ptr, fake_address3);
}

TEST_F(ArenaTest, GlobalArenaReleaseMergePreviousAndNext)  // NOLINT
{
  auto sblk = global_arena_->acquire(256);
  auto sb2  = global_arena_->acquire(1_KiB);
  auto sb3  = global_arena_->acquire(512);
  global_arena_->release(std::move(sblk));
  global_arena_->release(std::move(sb3));
  global_arena_->release(std::move(sb2));
  auto* ptr = global_arena_->allocate(arena_size_);
  EXPECT_EQ(ptr, fake_address3);
}

TEST_F(ArenaTest, GlobalArenaReleaseMultiple)  // NOLINT
{
  std::set<superblock> superblocks{};
  auto sblk = global_arena_->acquire(256);
  superblocks.insert(std::move(sblk));
  auto sb2 = global_arena_->acquire(1_KiB);
  superblocks.insert(std::move(sb2));
  auto sb3 = global_arena_->acquire(512);
  superblocks.insert(std::move(sb3));
  global_arena_->release(superblocks);
  auto* ptr = global_arena_->allocate(arena_size_);
  EXPECT_EQ(ptr, fake_address3);
}

TEST_F(ArenaTest, GlobalArenaAllocate)  // NOLINT
{
  auto* ptr = global_arena_->allocate(superblock::minimum_size * 2);
  EXPECT_EQ(ptr, fake_address3);
}

TEST_F(ArenaTest, GlobalArenaAllocateExtraLarge)  // NOLINT
{
  EXPECT_EQ(global_arena_->allocate(1_PiB), nullptr);
  EXPECT_EQ(global_arena_->allocate(1_PiB), nullptr);
}

TEST_F(ArenaTest, GlobalArenaDeallocate)  // NOLINT
{
  auto* ptr = global_arena_->allocate(superblock::minimum_size * 2);
  EXPECT_EQ(ptr, fake_address3);
  global_arena_->deallocate(ptr, superblock::minimum_size * 2, {});
  ptr = global_arena_->allocate(superblock::minimum_size * 2);
  EXPECT_EQ(ptr, fake_address3);
}

TEST_F(ArenaTest, GlobalArenaDeallocateAlignUp)  // NOLINT
{
  auto* ptr  = global_arena_->allocate(superblock::minimum_size + 256);
  auto* ptr2 = global_arena_->allocate(superblock::minimum_size + 512);
  global_arena_->deallocate(ptr, superblock::minimum_size + 256, {});
  global_arena_->deallocate(ptr2, superblock::minimum_size + 512, {});
  EXPECT_EQ(global_arena_->allocate(arena_size_), fake_address3);
}

TEST_F(ArenaTest, GlobalArenaDeallocateFromOtherArena)  // NOLINT
{
  auto sblk      = global_arena_->acquire(512);
  auto const blk = sblk.first_fit(512);
  global_arena_->release(std::move(sblk));
  global_arena_->deallocate(blk.pointer(), blk.size());
  EXPECT_EQ(global_arena_->allocate(arena_size_), fake_address3);
}

/**
 * Test arena.
 */

TEST_F(ArenaTest, ArenaAllocate)  // NOLINT
{
  EXPECT_EQ(arena_->allocate(superblock::minimum_size), fake_address3);
  EXPECT_EQ(arena_->allocate(256), fake_address4);
}

TEST_F(ArenaTest, ArenaDeallocate)  // NOLINT
{
  auto* ptr = arena_->allocate(superblock::minimum_size);
  arena_->deallocate(ptr, superblock::minimum_size, {});
  auto* ptr2 = arena_->allocate(256);
  arena_->deallocate(ptr2, 256, {});
  EXPECT_EQ(arena_->allocate(superblock::minimum_size), fake_address3);
}

TEST_F(ArenaTest, ArenaDeallocateMergePrevious)  // NOLINT
{
  auto* ptr  = arena_->allocate(256);
  auto* ptr2 = arena_->allocate(256);
  arena_->allocate(256);
  arena_->deallocate(ptr, 256, {});
  arena_->deallocate(ptr2, 256, {});
  EXPECT_EQ(arena_->allocate(512), fake_address3);
}

TEST_F(ArenaTest, ArenaDeallocateMergeNext)  // NOLINT
{
  auto* ptr  = arena_->allocate(256);
  auto* ptr2 = arena_->allocate(256);
  arena_->allocate(256);
  arena_->deallocate(ptr2, 256, {});
  arena_->deallocate(ptr, 256, {});
  EXPECT_EQ(arena_->allocate(512), fake_address3);
}

TEST_F(ArenaTest, ArenaDeallocateMergePreviousAndNext)  // NOLINT
{
  auto* ptr  = arena_->allocate(256);
  auto* ptr2 = arena_->allocate(256);
  arena_->deallocate(ptr, 256, {});
  arena_->deallocate(ptr2, 256, {});
  EXPECT_EQ(arena_->allocate(2_KiB), fake_address3);
}

TEST_F(ArenaTest, ArenaDefragment)  // NOLINT
{
  std::vector<void*> pointers;
  std::size_t num_pointers{4};
  for (std::size_t i = 0; i < num_pointers; i++) {
    pointers.push_back(arena_->allocate(superblock::minimum_size));
  }
  for (auto* ptr : pointers) {
    arena_->deallocate(ptr, superblock::minimum_size, {});
  }
  EXPECT_EQ(global_arena_->allocate(arena_size_), nullptr);
  arena_->defragment();
  EXPECT_EQ(global_arena_->allocate(arena_size_), fake_address3);
}

/**
 * Test arena_memory_resource.
 */

TEST_F(ArenaTest, ThrowOnNullUpstream)  // NOLINT
{
  auto construct_nullptr = []() { arena_mr mr{nullptr}; };
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto)
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);
}

TEST_F(ArenaTest, AllocateNinetyPercent)  // NOLINT
{
  EXPECT_NO_THROW([]() {  // NOLINT(cppcoreguidelines-avoid-goto)
    auto const free = rmm::detail::available_device_memory().first;
    auto const ninety_percent =
      rmm::detail::align_up(static_cast<std::size_t>(static_cast<double>(free) * 0.9),
                            rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
    arena_mr mr(rmm::mr::get_current_device_resource(), ninety_percent);
  }());
}

TEST_F(ArenaTest, SmallMediumLarge)  // NOLINT
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

TEST_F(ArenaTest, Defragment)  // NOLINT
{
  EXPECT_NO_THROW([]() {  // NOLINT(cppcoreguidelines-avoid-goto)
    auto const arena_size = superblock::minimum_size * 4;
    arena_mr mr(rmm::mr::get_current_device_resource(), arena_size);
    std::vector<std::thread> threads;
    std::size_t num_threads{4};
    threads.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
      threads.emplace_back(std::thread([&] {
        cuda_stream stream{};
        void* ptr = mr.allocate(32_KiB, stream);
        mr.deallocate(ptr, 32_KiB, stream);
      }));
    }
    for (auto& thread : threads) {
      thread.join();
    }

    auto* ptr = mr.allocate(arena_size);
    mr.deallocate(ptr, arena_size);
  }());
}

TEST_F(ArenaTest, DumpLogOnFailure)  // NOLINT
{
  arena_mr mr{rmm::mr::get_current_device_resource(), 1_MiB, true};

  {  // make the log interesting
    std::vector<std::thread> threads;
    std::size_t num_threads{4};
    threads.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
      threads.emplace_back(std::thread([&] {
        void* ptr = mr.allocate(32_KiB);
        mr.deallocate(ptr, 32_KiB);
      }));
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto)
  EXPECT_THROW(mr.allocate(8_MiB), rmm::out_of_memory);

  struct stat file_status {
  };
  EXPECT_EQ(stat("rmm_arena_memory_dump.log", &file_status), 0);
  EXPECT_GE(file_status.st_size, 0);
}

TEST_F(ArenaTest, FeatureSupport)  // NOLINT
{
  arena_mr mr{rmm::mr::get_current_device_resource(), 1_MiB};
  EXPECT_TRUE(mr.supports_streams());
  EXPECT_FALSE(mr.supports_get_mem_info());
  auto [free, total] = mr.get_mem_info(rmm::cuda_stream_default);
  EXPECT_EQ(free, 0);
  EXPECT_EQ(total, 0);
}

}  // namespace
}  // namespace rmm::test
