/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/logging_assert.hpp>
#include <rmm/logger.hpp>

#include <cuda_runtime_api.h>

#include <fmt/core.h>
#include <spdlog/common.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>

namespace rmm::mr::detail::arena {

/**
 * @brief Align up to nearest size class.
 *
 * @param[in] value value to align.
 * @return Return the aligned value.
 */
inline std::size_t align_to_size_class(std::size_t value) noexcept
{
  // See http://jemalloc.net/jemalloc.3.html.
  // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
  static std::array<std::size_t, 117> size_classes{
    // clang-format off
    // Spacing 256:
    256UL, 512UL, 768UL, 1024UL, 1280UL, 1536UL, 1792UL, 2048UL,
    // Spacing 512:
    2560UL, 3072UL, 3584UL, 4096UL,
    // Spacing 1 KiB:
    5UL << 10, 6UL << 10, 7UL << 10, 8UL << 10,
    // Spacing 2 KiB:
    10UL << 10, 12UL << 10, 14UL << 10, 16UL << 10,
    // Spacing 4 KiB:
    20UL << 10, 24UL << 10, 28UL << 10, 32UL << 10,
    // Spacing 8 KiB:
    40UL << 10, 48UL << 10, 54UL << 10, 64UL << 10,
    // Spacing 16 KiB:
    80UL << 10, 96UL << 10, 112UL << 10, 128UL << 10,
    // Spacing 32 KiB:
    160UL << 10, 192UL << 10, 224UL << 10, 256UL << 10,
    // Spacing 64 KiB:
    320UL << 10, 384UL << 10, 448UL << 10, 512UL << 10,
    // Spacing 128 KiB:
    640UL << 10, 768UL << 10, 896UL << 10, 1UL << 20,
    // Spacing 256 KiB:
    1280UL << 10, 1536UL << 10, 1792UL << 10, 2UL << 20,
    // Spacing 512 KiB:
    2560UL << 10, 3UL << 20, 3584UL << 10, 4UL << 20,
    // Spacing 1 MiB:
    5UL << 20, 6UL << 20, 7UL << 20, 8UL << 20,
    // Spacing 2 MiB:
    10UL << 20, 12UL << 20, 14UL << 20, 16UL << 20,
    // Spacing 4 MiB:
    20UL << 20, 24UL << 20, 28UL << 20, 32UL << 20,
    // Spacing 8 MiB:
    40UL << 20, 48UL << 20, 56UL << 20, 64UL << 20,
    // Spacing 16 MiB:
    80UL << 20, 96UL << 20, 112UL << 20, 128UL << 20,
    // Spacing 32 MiB:
    160UL << 20, 192UL << 20, 224UL << 20, 256UL << 20,
    // Spacing 64 MiB:
    320UL << 20, 384UL << 20, 448UL << 20, 512UL << 20,
    // Spacing 128 MiB:
    640UL << 20, 768UL << 20, 896UL << 20, 1UL << 30,
    // Spacing 256 MiB:
    1280UL << 20, 1536UL << 20, 1792UL << 20, 2UL << 30,
    // Spacing 512 MiB:
    2560UL << 20, 3UL << 30, 3584UL << 20, 4UL << 30,
    // Spacing 1 GiB:
    5UL << 30, 6UL << 30, 7UL << 30, 8UL << 30,
    // Spacing 2 GiB:
    10UL << 30, 12UL << 30, 14UL << 30, 16UL << 30,
    // Spacing 4 GiB:
    20UL << 30, 24UL << 30, 28UL << 30, 32UL << 30,
    // Spacing 8 GiB:
    40UL << 30, 48UL << 30, 56UL << 30, 64UL << 30,
    // Spacing 16 GiB:
    80UL << 30, 96UL << 30, 112UL << 30, 128UL << 30,
    // Spacing 32 Gib:
    160UL << 30, 192UL << 30, 224UL << 30, 256UL << 30,
    // Catch all:
    std::numeric_limits<std::size_t>::max()
    // clang-format on
  };
  // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)

  auto* bound = std::lower_bound(size_classes.begin(), size_classes.end(), value);
  RMM_LOGGING_ASSERT(bound != size_classes.end());
  return *bound;
}

/**
 * @brief Represents a contiguous region of memory.
 */
class byte_span {
 public:
  /**
   * @brief Construct a default span.
   */
  byte_span() = default;

  /**
   * @brief Construct a span given a pointer and size.
   *
   * @param pointer The address for the beginning of the span.
   * @param size The size of the span.
   */
  byte_span(void* pointer, std::size_t size) : pointer_{static_cast<char*>(pointer)}, size_{size}
  {
    RMM_LOGGING_ASSERT(pointer != nullptr);
    RMM_LOGGING_ASSERT(size > 0);
  }

  /// Returns the underlying pointer.
  [[nodiscard]] char* pointer() const { return pointer_; }

  /// Returns the size of the span.
  [[nodiscard]] std::size_t size() const { return size_; }

  /// Returns the end of the span.
  [[nodiscard]] char* end() const
  {
    return pointer_ + size_;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  }

  /// Returns true if this span is valid (non-null), false otherwise.
  [[nodiscard]] bool is_valid() const { return pointer_ != nullptr && size_ > 0; }

  /// Used by std::set to compare spans.
  bool operator<(byte_span const& span) const
  {
    RMM_LOGGING_ASSERT(span.is_valid());
    return pointer_ < span.pointer_;
  }

 private:
  char* pointer_{};     ///< Raw memory pointer.
  std::size_t size_{};  ///< Size in bytes.
};

/// Calculate the total size of a set of spans.
template <typename T>
inline auto total_memory_size(std::set<T> const& spans)
{
  return std::accumulate(
    spans.cbegin(), spans.cend(), std::size_t{}, [](auto const& lhs, auto const& rhs) {
      return lhs + rhs.size();
    });
}

/**
 * @brief Represents a chunk of memory that can be allocated and deallocated.
 */
class block final : public byte_span {
 public:
  using byte_span::byte_span;

  /**
   * @brief Is this block large enough to fit `bytes` bytes?
   *
   * @param bytes The size in bytes to check for fit.
   * @return true if this block is at least `bytes` bytes.
   */
  [[nodiscard]] bool fits(std::size_t bytes) const
  {
    RMM_LOGGING_ASSERT(is_valid());
    RMM_LOGGING_ASSERT(bytes > 0);
    return size() >= bytes;
  }

  /**
   * @brief Verifies whether this block can be merged to the beginning of block blk.
   *
   * @param blk The block to check for contiguity.
   * @return true Returns true if this block's `pointer` + `size` == `blk.pointer`.
   */
  [[nodiscard]] bool is_contiguous_before(block const& blk) const
  {
    RMM_LOGGING_ASSERT(is_valid());
    RMM_LOGGING_ASSERT(blk.is_valid());
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return pointer() + size() == blk.pointer();
  }

  /**
   * @brief Split this block into two by the given size.
   *
   * @param bytes The size in bytes of the first block.
   * @return std::pair<block, block> A pair of blocks split by bytes.
   */
  [[nodiscard]] std::pair<block, block> split(std::size_t bytes) const
  {
    RMM_LOGGING_ASSERT(is_valid());
    RMM_LOGGING_ASSERT(size() > bytes);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return {{pointer(), bytes}, {pointer() + bytes, size() - bytes}};
  }

  /**
   * @brief Coalesce two contiguous blocks into one.
   *
   * `this->is_contiguous_before(blk)` must be true.
   *
   * @param blk block to merge.
   * @return block The merged block.
   */
  [[nodiscard]] block merge(block const& blk) const
  {
    RMM_LOGGING_ASSERT(is_contiguous_before(blk));
    return {pointer(), size() + blk.size()};
  }
};

/// Comparison function for block sizes.
inline bool block_size_compare(block const& lhs, block const& rhs)
{
  RMM_LOGGING_ASSERT(lhs.is_valid());
  RMM_LOGGING_ASSERT(rhs.is_valid());
  return lhs.size() < rhs.size();
}

/**
 * @brief Represents a large chunk of memory that is exchanged between the global arena and
 * per-thread arenas.
 */
class superblock final : public byte_span {
 public:
  /// Minimum size of a superblock (1 MiB).
  static constexpr std::size_t minimum_size{1UL << 20};
  /// Maximum size of a superblock (1 TiB), as a sanity check.
  static constexpr std::size_t maximum_size{1UL << 40};

  /**
   * @brief Construct a default superblock.
   */
  superblock() = default;

  /**
   * @brief Construct a superblock given a pointer and size.
   *
   * @param pointer The address for the beginning of the superblock.
   * @param size The size of the superblock.
   */
  superblock(void* pointer, std::size_t size) : byte_span{pointer, size}
  {
    RMM_LOGGING_ASSERT(size >= minimum_size);
    RMM_LOGGING_ASSERT(size <= maximum_size);
    free_blocks_.emplace(pointer, size);
  }

  // Disable copy semantics.
  superblock(superblock const&)            = delete;
  superblock& operator=(superblock const&) = delete;
  // Allow move semantics.
  superblock(superblock&&) noexcept            = default;
  superblock& operator=(superblock&&) noexcept = default;

  ~superblock() = default;

  /**
   * @brief Is this superblock empty?
   *
   * @return true if this superblock is empty.
   */
  [[nodiscard]] bool empty() const
  {
    RMM_LOGGING_ASSERT(is_valid());
    return free_blocks_.size() == 1 && free_blocks_.cbegin()->size() == size();
  }

  /**
   * @brief Return the number of free blocks.
   *
   * @return the number of free blocks.
   */
  [[nodiscard]] std::size_t free_blocks() const
  {
    RMM_LOGGING_ASSERT(is_valid());
    return free_blocks_.size();
  }

  /**
   * @brief Whether this superblock contains the given block.
   *
   * @param blk The block to search for.
   * @return true if the given block belongs to this superblock.
   */
  [[nodiscard]] bool contains(block const& blk) const
  {
    RMM_LOGGING_ASSERT(is_valid());
    RMM_LOGGING_ASSERT(blk.is_valid());
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return pointer() <= blk.pointer() && pointer() + size() >= blk.pointer() + blk.size();
  }

  /**
   * @brief Can this superblock fit `bytes` bytes?
   *
   * @param bytes The size in bytes to check for fit.
   * @return true if this superblock can fit `bytes` bytes.
   */
  [[nodiscard]] bool fits(std::size_t bytes) const
  {
    RMM_LOGGING_ASSERT(is_valid());
    return std::any_of(free_blocks_.cbegin(), free_blocks_.cend(), [bytes](auto const& blk) {
      return blk.fits(bytes);
    });
  }

  /**
   * @brief Verifies whether this superblock can be merged to the beginning of superblock s.
   *
   * @param s The superblock to check for contiguity.
   * @return true Returns true if both superblocks are empty and this superblock's
   * `pointer` + `size` == `s.ptr`.
   */
  [[nodiscard]] bool is_contiguous_before(superblock const& sblk) const
  {
    RMM_LOGGING_ASSERT(is_valid());
    RMM_LOGGING_ASSERT(sblk.is_valid());
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return empty() && sblk.empty() && pointer() + size() == sblk.pointer();
  }

  /**
   * @brief Split this superblock into two by the given size.
   *
   * @param bytes The size in bytes of the first block.
   * @return superblock_pair A pair of superblocks split by bytes.
   */
  [[nodiscard]] std::pair<superblock, superblock> split(std::size_t bytes) const
  {
    RMM_LOGGING_ASSERT(is_valid());
    RMM_LOGGING_ASSERT(empty() && bytes >= minimum_size && size() >= bytes + minimum_size);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return {superblock{pointer(), bytes}, superblock{pointer() + bytes, size() - bytes}};
  }

  /**
   * @brief Coalesce two contiguous superblocks into one.
   *
   * `this->is_contiguous_before(s)` must be true.
   *
   * @param sblk superblock to merge.
   * @return block The merged block.
   */
  [[nodiscard]] superblock merge(superblock const& sblk) const
  {
    RMM_LOGGING_ASSERT(is_contiguous_before(sblk));
    return {pointer(), size() + sblk.size()};
  }

  /**
   * @brief Get the first free block of at least `size` bytes.
   *
   * @param size The number of bytes to allocate.
   * @return block A block of memory of at least `size` bytes, or an empty block if not found.
   */
  block first_fit(std::size_t size)
  {
    RMM_LOGGING_ASSERT(is_valid());
    RMM_LOGGING_ASSERT(size > 0);

    auto fits       = [size](auto const& blk) { return blk.fits(size); };
    auto const iter = std::find_if(free_blocks_.cbegin(), free_blocks_.cend(), fits);
    if (iter == free_blocks_.cend()) { return {}; }

    // Remove the block from the free list.
    auto const blk  = *iter;
    auto const next = free_blocks_.erase(iter);

    if (blk.size() > size) {
      // Split the block and put the remainder back.
      auto const split = blk.split(size);
      free_blocks_.insert(next, split.second);
      return split.first;
    }
    return blk;
  }

  /**
   * @brief Coalesce the given block with other free blocks.
   *
   * @param blk The block to coalesce.
   */
  void coalesce(block const& blk)  // NOLINT(readability-function-cognitive-complexity)
  {
    RMM_LOGGING_ASSERT(is_valid());
    RMM_LOGGING_ASSERT(blk.is_valid());
    RMM_LOGGING_ASSERT(contains(blk));

    // Find the right place (in ascending address order) to insert the block.
    auto const next     = free_blocks_.lower_bound(blk);
    auto const previous = next == free_blocks_.cbegin() ? next : std::prev(next);

    // Coalesce with neighboring blocks.
    bool const merge_prev = previous != free_blocks_.cend() && previous->is_contiguous_before(blk);
    bool const merge_next = next != free_blocks_.cend() && blk.is_contiguous_before(*next);

    if (merge_prev && merge_next) {
      auto const merged = previous->merge(blk).merge(*next);
      free_blocks_.erase(previous);
      auto const iter = free_blocks_.erase(next);
      free_blocks_.insert(iter, merged);
    } else if (merge_prev) {
      auto const merged = previous->merge(blk);
      auto const iter   = free_blocks_.erase(previous);
      free_blocks_.insert(iter, merged);
    } else if (merge_next) {
      auto const merged = blk.merge(*next);
      auto const iter   = free_blocks_.erase(next);
      free_blocks_.insert(iter, merged);
    } else {
      free_blocks_.insert(next, blk);
    }
  }

  /**
   * @brief Find the total free block size.
   * @return the total free block size.
   */
  [[nodiscard]] std::size_t total_free_size() const { return total_memory_size(free_blocks_); }

  /**
   * @brief Find the max free block size.
   * @return the max free block size.
   */
  [[nodiscard]] std::size_t max_free_size() const
  {
    if (free_blocks_.empty()) { return 0; }
    return std::max_element(free_blocks_.cbegin(), free_blocks_.cend(), block_size_compare)->size();
  }

 private:
  /// Address-ordered set of free blocks.
  std::set<block> free_blocks_{};
};

/// Calculate the total free size of a set of superblocks.
inline auto total_free_size(std::set<superblock> const& superblocks)
{
  return std::accumulate(
    superblocks.cbegin(), superblocks.cend(), std::size_t{}, [](auto const& lhs, auto const& rhs) {
      return lhs + rhs.total_free_size();
    });
}

/// Find the max free size from a set of superblocks.
inline auto max_free_size(std::set<superblock> const& superblocks)
{
  std::size_t size{};
  for (auto const& sblk : superblocks) {
    size = std::max(size, sblk.max_free_size());
  }
  return size;
};

/**
 * @brief The global arena for allocating memory from the upstream memory resource.
 *
 * The global arena is a shared memory pool from which other arenas allocate superblocks.
 *
 * @tparam Upstream Memory resource to use for allocating the arena. Implements
 * rmm::mr::device_memory_resource interface.
 */
template <typename Upstream>
class global_arena final {
 public:
  /**
   * @brief Construct a global arena.
   *
   * @throws rmm::logic_error if `upstream_mr == nullptr`.
   *
   * @param upstream_mr The memory resource from which to allocate blocks for the pool
   * @param arena_size Size in bytes of the global arena. Defaults to half of the available memory
   * on the current device.
   */
  global_arena(Upstream* upstream_mr, std::optional<std::size_t> arena_size)
    : upstream_mr_{upstream_mr}
  {
    RMM_EXPECTS(nullptr != upstream_mr_, "Unexpected null upstream pointer.");
    auto const size =
      rmm::align_down(arena_size.value_or(default_size()), rmm::CUDA_ALLOCATION_ALIGNMENT);
    RMM_EXPECTS(size >= superblock::minimum_size,
                "Arena size smaller than minimum superblock size.");
    initialize(size);
  }

  // Disable copy (and move) semantics.
  global_arena(global_arena const&)                = delete;
  global_arena& operator=(global_arena const&)     = delete;
  global_arena(global_arena&&) noexcept            = delete;
  global_arena& operator=(global_arena&&) noexcept = delete;

  /**
   * @brief Destroy the global arena and deallocate all memory it allocated using the upstream
   * resource.
   */
  ~global_arena()
  {
    std::lock_guard lock(mtx_);
    upstream_mr_->deallocate(upstream_block_.pointer(), upstream_block_.size());
  }

  /**
   * @brief Should allocation of `size` bytes be handled by the global arena directly?
   *
   * @param size The size in bytes of the allocation.
   * @return bool True if the allocation should be handled by the global arena.
   */
  bool handles(std::size_t size) const { return size > superblock::minimum_size; }

  /**
   * @brief Acquire a superblock that can fit a block of the given size.
   *
   * @param size The size in bytes of the allocation.
   * @return superblock The acquired superblock.
   */
  superblock acquire(std::size_t size)
  {
    // Superblocks should only be acquired if the size is not directly handled by the global arena.
    RMM_LOGGING_ASSERT(!handles(size));
    std::lock_guard lock(mtx_);
    return first_fit(size);
  }

  /**
   * @brief Release a superblock.
   *
   * @param s Superblock to be released.
   */
  void release(superblock&& sblk)
  {
    RMM_LOGGING_ASSERT(sblk.is_valid());
    std::lock_guard lock(mtx_);
    coalesce(std::move(sblk));
  }

  /**
   * @brief Release a set of superblocks from a dying arena.
   *
   * @param superblocks The set of superblocks.
   */
  void release(std::set<superblock>& superblocks)
  {
    std::lock_guard lock(mtx_);
    while (!superblocks.empty()) {
      auto sblk = std::move(superblocks.extract(superblocks.cbegin()).value());
      RMM_LOGGING_ASSERT(sblk.is_valid());
      coalesce(std::move(sblk));
    }
  }

  /**
   * @brief Allocate a large block directly.
   *
   * @param size The size in bytes of the allocation.
   * @return void* Pointer to the newly allocated memory.
   */
  void* allocate(std::size_t size)
  {
    RMM_LOGGING_ASSERT(handles(size));
    std::lock_guard lock(mtx_);
    auto sblk = first_fit(size);
    if (sblk.is_valid()) {
      auto blk = sblk.first_fit(size);
      superblocks_.insert(std::move(sblk));
      return blk.pointer();
    }
    return nullptr;
  }

  /**
   * @brief Deallocate memory pointed to by `ptr`.
   *
   * @param ptr Pointer to be deallocated.
   * @param size The size in bytes of the allocation. This must be equal to the value of `size`
   * that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation.
   * @return bool true if the allocation is found, false otherwise.
   */
  bool deallocate(void* ptr, std::size_t size, cuda_stream_view stream)
  {
    RMM_LOGGING_ASSERT(handles(size));
    stream.synchronize_no_throw();
    return deallocate(ptr, size);
  }

  /**
   * @brief Deallocate memory pointed to by `ptr`.
   *
   * @param ptr Pointer to be deallocated.
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `ptr`.
   * @return bool true if the allocation is found, false otherwise.
   */
  bool deallocate(void* ptr, std::size_t bytes)
  {
    std::lock_guard lock(mtx_);

    block const blk{ptr, bytes};
    auto const iter = std::find_if(superblocks_.cbegin(),
                                   superblocks_.cend(),
                                   [&](auto const& sblk) { return sblk.contains(blk); });
    if (iter == superblocks_.cend()) { return false; }

    auto sblk = std::move(superblocks_.extract(iter).value());
    sblk.coalesce(blk);
    if (sblk.empty()) {
      coalesce(std::move(sblk));
    } else {
      superblocks_.insert(std::move(sblk));
    }
    return true;
  }

  /**
   * @brief Dump memory to log.
   *
   * @param logger the spdlog logger to use
   */
  void dump_memory_log(std::shared_ptr<spdlog::logger> const& logger) const
  {
    std::lock_guard lock(mtx_);

    logger->info("  Arena size: {}", rmm::detail::bytes{upstream_block_.size()});
    logger->info("  # superblocks: {}", superblocks_.size());
    if (!superblocks_.empty()) {
      logger->debug("  Total size of superblocks: {}",
                    rmm::detail::bytes{total_memory_size(superblocks_)});
      auto const total_free    = total_free_size(superblocks_);
      auto const max_free      = max_free_size(superblocks_);
      auto const fragmentation = (1 - max_free / static_cast<double>(total_free)) * 100;
      logger->info("  Total free memory: {}", rmm::detail::bytes{total_free});
      logger->info("  Largest block of free memory: {}", rmm::detail::bytes{max_free});
      logger->info("  Fragmentation: {:.2f}%", fragmentation);

      auto index = 0;
      char* prev_end{};
      for (auto const& sblk : superblocks_) {
        if (prev_end == nullptr) { prev_end = sblk.pointer(); }
        logger->debug(
          "    Superblock {}: start={}, end={}, size={}, empty={}, # free blocks={}, max free={}, "
          "gap={}",
          index,
          fmt::ptr(sblk.pointer()),
          fmt::ptr(sblk.end()),
          rmm::detail::bytes{sblk.size()},
          sblk.empty(),
          sblk.free_blocks(),
          rmm::detail::bytes{sblk.max_free_size()},
          rmm::detail::bytes{static_cast<size_t>(sblk.pointer() - prev_end)});
        prev_end = sblk.end();
        index++;
      }
    }
  }

 private:
  /**
   * @brief Default size of the global arena if unspecified.
   * @return the default global arena size.
   */
  constexpr std::size_t default_size() const
  {
    auto const [free, total] = rmm::available_device_memory();
    return free / 2;
  }

  /**
   * @brief Allocate space from upstream to initialize the arena.
   *
   * @param size The size to allocate.
   */
  void initialize(std::size_t size)
  {
    upstream_block_ = {upstream_mr_->allocate(size), size};
    superblocks_.emplace(upstream_block_.pointer(), size);
  }

  /**
   * @brief Get the first superblock that can fit a block of at least `size` bytes.
   *
   * Address-ordered first-fit has shown to perform slightly better than best-fit when it comes to
   * memory fragmentation, and slightly cheaper to implement. It is also used by some popular
   * allocators such as jemalloc.
   *
   * \see Johnstone, M. S., & Wilson, P. R. (1998). The memory fragmentation problem: Solved?. ACM
   * Sigplan Notices, 34(3), 26-36.
   *
   * @param size The number of bytes to allocate.
   * @param minimum_size The minimum size of the superblock required.
   * @return superblock A superblock that can fit at least `size` bytes, or empty if not found.
   */
  superblock first_fit(std::size_t size)
  {
    auto const iter = std::find_if(superblocks_.cbegin(),
                                   superblocks_.cend(),
                                   [=](auto const& sblk) { return sblk.fits(size); });
    if (iter == superblocks_.cend()) { return {}; }

    auto sblk           = std::move(superblocks_.extract(iter).value());
    auto const min_size = std::max(superblock::minimum_size, size);
    if (sblk.empty() && sblk.size() >= min_size + superblock::minimum_size) {
      // Split the superblock and put the remainder back.
      auto [head, tail] = sblk.split(min_size);
      superblocks_.insert(std::move(tail));
      return std::move(head);
    }
    return sblk;
  }

  /**
   * @brief Coalesce the given superblock with other empty superblocks.
   *
   * @param sblk The superblock to coalesce.
   */
  void coalesce(superblock&& sblk)
  {
    RMM_LOGGING_ASSERT(sblk.is_valid());

    // Find the right place (in ascending address order) to insert the block.
    auto const next     = superblocks_.lower_bound(sblk);
    auto const previous = next == superblocks_.cbegin() ? next : std::prev(next);

    // Coalesce with neighboring blocks.
    bool const merge_prev = previous != superblocks_.cend() && previous->is_contiguous_before(sblk);
    bool const merge_next = next != superblocks_.cend() && sblk.is_contiguous_before(*next);

    if (merge_prev && merge_next) {
      auto prev_sb = std::move(superblocks_.extract(previous).value());
      auto next_sb = std::move(superblocks_.extract(next).value());
      auto merged  = prev_sb.merge(sblk).merge(next_sb);
      superblocks_.insert(std::move(merged));
    } else if (merge_prev) {
      auto prev_sb = std::move(superblocks_.extract(previous).value());
      auto merged  = prev_sb.merge(sblk);
      superblocks_.insert(std::move(merged));
    } else if (merge_next) {
      auto next_sb = std::move(superblocks_.extract(next).value());
      auto merged  = sblk.merge(next_sb);
      superblocks_.insert(std::move(merged));
    } else {
      superblocks_.insert(std::move(sblk));
    }
  }

  /// The upstream resource to allocate memory from.
  Upstream* upstream_mr_;
  /// Block allocated from upstream so that it can be quickly freed.
  block upstream_block_;
  /// Address-ordered set of superblocks.
  std::set<superblock> superblocks_;
  /// Mutex for exclusive lock.
  mutable std::mutex mtx_;
};

/**
 * @brief An arena for allocating memory for a thread.
 *
 * An arena is a per-thread or per-non-default-stream memory pool. It allocates
 * superblocks from the global arena, and returns them when the superblocks become empty.
 *
 * @tparam Upstream Memory resource to use for allocating the global arena. Implements
 * rmm::mr::device_memory_resource interface.
 */
template <typename Upstream>
class arena {
 public:
  /**
   * @brief Construct an `arena`.
   *
   * @param global_arena The global arena from which to allocate superblocks.
   */
  explicit arena(global_arena<Upstream>& global_arena) : global_arena_{global_arena} {}

  // Disable copy (and move) semantics.
  arena(arena const&)                = delete;
  arena& operator=(arena const&)     = delete;
  arena(arena&&) noexcept            = delete;
  arena& operator=(arena&&) noexcept = delete;

  ~arena() = default;

  /**
   * @brief Allocates memory of size at least `size` bytes.
   *
   * @param size The size in bytes of the allocation.
   * @return void* Pointer to the newly allocated memory.
   */
  void* allocate(std::size_t size)
  {
    if (global_arena_.handles(size)) { return global_arena_.allocate(size); }
    std::lock_guard lock(mtx_);
    return get_block(size).pointer();
  }

  /**
   * @brief Deallocate memory pointed to by `ptr`, and possibly return superblocks to upstream.
   *
   * @param ptr Pointer to be deallocated.
   * @param size The size in bytes of the allocation. This must be equal to the value of `size`
   * that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation.
   * @return bool true if the allocation is found, false otherwise.
   */
  bool deallocate(void* ptr, std::size_t size, cuda_stream_view stream)
  {
    if (global_arena_.handles(size) && global_arena_.deallocate(ptr, size, stream)) { return true; }
    return deallocate(ptr, size);
  }

  /**
   * @brief Deallocate memory pointed to by `ptr`, and possibly return superblocks to upstream.
   *
   * @param ptr Pointer to be deallocated.
   * @param size The size in bytes of the allocation. This must be equal to the value of `size`
   * that was passed to the `allocate` call that returned `p`.
   * @return bool true if the allocation is found, false otherwise.
   */
  bool deallocate(void* ptr, std::size_t size)
  {
    std::lock_guard lock(mtx_);
    return deallocate_from_superblock({ptr, size});
  }

  /**
   * @brief Clean the arena and release all superblocks to the global arena.
   */
  void clean()
  {
    std::lock_guard lock(mtx_);
    global_arena_.release(superblocks_);
    superblocks_.clear();
  }

  /**
   * @brief Defragment the arena and release empty superblock to the global arena.
   */
  void defragment()
  {
    std::lock_guard lock(mtx_);
    while (true) {
      auto const iter = std::find_if(
        superblocks_.cbegin(), superblocks_.cend(), [](auto const& sblk) { return sblk.empty(); });
      if (iter == superblocks_.cend()) { return; }
      global_arena_.release(std::move(superblocks_.extract(iter).value()));
    }
  }

 private:
  /**
   * @brief Get an available memory block of at least `size` bytes.
   *
   * @param size The number of bytes to allocate.
   * @return A block of memory of at least `size` bytes.
   */
  block get_block(std::size_t size)
  {
    // Find the first-fit free block.
    auto const blk = first_fit(size);
    if (blk.is_valid()) { return blk; }

    // No existing larger blocks available, so grow the arena and obtain a superblock.
    return expand_arena(size);
  }

  /**
   * @brief Get the first free block of at least `size` bytes.
   *
   * Address-ordered first-fit has shown to perform slightly better than best-fit when it comes to
   * memory fragmentation, and slightly cheaper to implement. It is also used by some popular
   * allocators such as jemalloc.
   *
   * \see Johnstone, M. S., & Wilson, P. R. (1998). The memory fragmentation problem: Solved?. ACM
   * Sigplan Notices, 34(3), 26-36.
   *
   * @param size The number of bytes to allocate.
   * @return block A block of memory of at least `size` bytes, or an empty block if not found.
   */
  block first_fit(std::size_t size)
  {
    auto const iter = std::find_if(superblocks_.cbegin(),
                                   superblocks_.cend(),
                                   [size](auto const& sblk) { return sblk.fits(size); });
    if (iter == superblocks_.cend()) { return {}; }

    auto sblk      = std::move(superblocks_.extract(iter).value());
    auto const blk = sblk.first_fit(size);
    superblocks_.insert(std::move(sblk));
    return blk;
  }

  /**
   * @brief Deallocate a block from the superblock it belongs to.
   *
   * @param blk The block to deallocate.
   * @param stream The stream to use for deallocation.
   * @return true if the block is found.
   */
  bool deallocate_from_superblock(block const& blk)
  {
    auto const iter = std::find_if(superblocks_.cbegin(),
                                   superblocks_.cend(),
                                   [&](auto const& sblk) { return sblk.contains(blk); });
    if (iter == superblocks_.cend()) { return false; }

    auto sblk = std::move(superblocks_.extract(iter).value());
    sblk.coalesce(blk);
    superblocks_.insert(std::move(sblk));
    return true;
  }

  /**
   * @brief Allocate space from upstream to supply the arena and return a block.
   *
   * @param size The number of bytes to allocate.
   * @return block A block of memory of at least `size` bytes.
   */
  block expand_arena(std::size_t size)
  {
    auto sblk = global_arena_.acquire(size);
    if (sblk.is_valid()) {
      RMM_LOGGING_ASSERT(sblk.size() >= superblock::minimum_size);
      auto const blk = sblk.first_fit(size);
      superblocks_.insert(std::move(sblk));
      return blk;
    }
    return {};
  }

  /// The global arena to allocate superblocks from.
  global_arena<Upstream>& global_arena_;
  /// Acquired superblocks.
  std::set<superblock> superblocks_;
  /// Mutex for exclusive lock.
  mutable std::mutex mtx_;
};

/**
 * @brief RAII-style cleaner for an arena.
 *
 * This is useful when a thread is about to terminate, and it contains a per-thread arena.
 *
 * @tparam Upstream Memory resource to use for allocating the global arena. Implements
 * rmm::mr::device_memory_resource interface.
 */
template <typename Upstream>
class arena_cleaner {
 public:
  explicit arena_cleaner(std::shared_ptr<arena<Upstream>> const& arena) : arena_(arena) {}

  // Disable copy (and move) semantics.
  arena_cleaner(arena_cleaner const&)            = delete;
  arena_cleaner& operator=(arena_cleaner const&) = delete;
  arena_cleaner(arena_cleaner&&) noexcept        = delete;
  arena_cleaner& operator=(arena_cleaner&&)      = delete;

  ~arena_cleaner()
  {
    if (!arena_.expired()) {
      auto arena_ptr = arena_.lock();
      arena_ptr->clean();
    }
  }

 private:
  /// A non-owning pointer to the arena that may need cleaning.
  std::weak_ptr<arena<Upstream>> arena_;
};

}  // namespace rmm::mr::detail::arena
