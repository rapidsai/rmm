/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/logger.hpp>

#include <cuda_runtime_api.h>

#include <spdlog/common.h>
#include <spdlog/fmt/bundled/ostream.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <unordered_map>

namespace rmm::mr::detail::arena {

/// Minimum size of a superblock (256 KiB).
constexpr std::size_t minimum_superblock_size = 1U << 18U;

/**
 * @brief Represents a chunk of memory that can be allocated and deallocated.
 *
 * A block bigger than a certain size is called a "superblock".
 */
class block {
 public:
  /**
   * @brief Construct a default block.
   */
  block() = default;

  /**
   * @brief Construct a block given a pointer and size.
   *
   * @param pointer The address for the beginning of the block.
   * @param size The size of the block.
   */
  block(char* pointer, std::size_t size) : pointer_(pointer), size_(size) {}

  /**
   * @brief Construct a block given a void pointer and size.
   *
   * @param pointer The address for the beginning of the block.
   * @param size The size of the block.
   */
  block(void* pointer, std::size_t size) : pointer_(static_cast<char*>(pointer)), size_(size) {}

  /// Returns the underlying pointer.
  [[nodiscard]] void* pointer() const { return pointer_; }

  /// Returns the size of the block.
  [[nodiscard]] std::size_t size() const { return size_; }

  /// Returns true if this block is valid (non-null), false otherwise.
  [[nodiscard]] bool is_valid() const { return pointer_ != nullptr; }

  /// Returns true if this block is a superblock, false otherwise.
  [[nodiscard]] bool is_superblock() const { return size_ >= minimum_superblock_size; }

  /**
   * @brief Verifies whether this block can be merged to the beginning of block b.
   *
   * @param b The block to check for contiguity.
   * @return true Returns true if this block's `pointer` + `size` == `b.ptr`, and `not b.is_head`,
                  false otherwise.
   */
  [[nodiscard]] bool is_contiguous_before(block const& blk) const
  {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return pointer_ + size_ == blk.pointer_;
  }

  /**
   * @brief Is this block large enough to fit `sz` bytes?
   *
   * @param size The size in bytes to check for fit.
   * @return true if this block is at least `sz` bytes.
   */
  [[nodiscard]] bool fits(std::size_t size) const { return size_ >= size; }

  /**
   * @brief Split this block into two by the given size.
   *
   * @param size The size in bytes of the first block.
   * @return std::pair<block, block> A pair of blocks split by sz.
   */
  [[nodiscard]] std::pair<block, block> split(std::size_t size) const
  {
    RMM_LOGGING_ASSERT(size_ >= size);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    if (size_ > size) { return {{pointer_, size}, {pointer_ + size, size_ - size}}; }
    return {*this, {}};
  }

  /**
   * @brief Coalesce two contiguous blocks into one.
   *
   * `this->is_contiguous_before(b)` must be true.
   *
   * @param b block to merge.
   * @return block The merged block.
   */
  [[nodiscard]] block merge(block const& blk) const
  {
    RMM_LOGGING_ASSERT(is_contiguous_before(blk));
    return {pointer_, size_ + blk.size_};
  }

  /// Used by std::set to compare blocks.
  bool operator<(block const& blk) const { return pointer_ < blk.pointer_; }

 private:
  char* pointer_{};     ///< Raw memory pointer.
  std::size_t size_{};  ///< Size in bytes.
};

inline bool block_size_compare(block lhs, block rhs) { return lhs.size() < rhs.size(); }

/**
 * @brief Align up to the allocation alignment.
 *
 * @param[in] v value to align
 * @return Return the aligned value
 */
constexpr std::size_t align_up(std::size_t value) noexcept
{
  return rmm::detail::align_up(value, rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
}

/**
 * @brief Align down to the allocation alignment.
 *
 * @param[in] v value to align
 * @return Return the aligned value
 */
constexpr std::size_t align_down(std::size_t value) noexcept
{
  return rmm::detail::align_down(value, rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
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
 * @param free_blocks The address-ordered set of free blocks.
 * @param size The number of bytes to allocate.
 * @return block A block of memory of at least `size` bytes, or an empty block if not found.
 */
inline block first_fit(std::set<block>& free_blocks, std::size_t size)
{
  auto const iter = std::find_if(
    free_blocks.cbegin(), free_blocks.cend(), [size](auto const& blk) { return blk.fits(size); });

  if (iter == free_blocks.cend()) { return {}; }
  // Remove the block from the free_list.
  auto const blk  = *iter;
  auto const next = free_blocks.erase(iter);

  if (blk.size() > size) {
    // Split the block and put the remainder back.
    auto const split = blk.split(size);
    free_blocks.insert(next, split.second);
    return split.first;
  }
  return blk;
}

/**
 * @brief Coalesce the given block with other free blocks.
 *
 * @param free_blocks The address-ordered set of free blocks.
 * @param b The block to coalesce.
 * @return block The coalesced block.
 */
inline block coalesce_block(std::set<block>& free_blocks, block const& blk)
{
  if (!blk.is_valid()) { return blk; }

  // Find the right place (in ascending address order) to insert the block.
  auto const next     = free_blocks.lower_bound(blk);
  auto const previous = next == free_blocks.cbegin() ? next : std::prev(next);

  // Coalesce with neighboring blocks.
  bool const merge_prev = previous->is_contiguous_before(blk);
  bool const merge_next = next != free_blocks.cend() && blk.is_contiguous_before(*next);

  block merged{};
  if (merge_prev && merge_next) {
    merged = previous->merge(blk).merge(*next);
    free_blocks.erase(previous);
    auto const iter = free_blocks.erase(next);
    free_blocks.insert(iter, merged);
  } else if (merge_prev) {
    merged          = previous->merge(blk);
    auto const iter = free_blocks.erase(previous);
    free_blocks.insert(iter, merged);
  } else if (merge_next) {
    merged          = blk.merge(*next);
    auto const iter = free_blocks.erase(next);
    free_blocks.insert(iter, merged);
  } else {
    free_blocks.emplace(blk);
    merged = blk;
  }
  return merged;
}

template <typename T>
inline auto total_block_size(T const& blocks)
{
  return std::accumulate(blocks.cbegin(), blocks.cend(), std::size_t{}, [](auto lhs, auto rhs) {
    return lhs + rhs.size();
  });
}

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
  /// The default initial size for the global arena.
  static constexpr std::size_t default_initial_size = std::numeric_limits<std::size_t>::max();
  /// The default maximum size for the global arena.
  static constexpr std::size_t default_maximum_size = std::numeric_limits<std::size_t>::max();
  /// Reserved memory that should not be allocated (64 MiB).
  static constexpr std::size_t reserved_size = 1U << 26U;

  /**
   * @brief Construct a global arena.
   *
   * @throws rmm::logic_error if `upstream_mr == nullptr`.
   * @throws rmm::logic_error if `initial_size` is neither the default nor aligned to a multiple of
   * 256 bytes.
   * @throws rmm::logic_error if `maximum_size` is neither the default nor aligned to a multiple of
   * 256 bytes.
   *
   * @param upstream_mr The memory resource from which to allocate blocks for the pool
   * @param initial_size Minimum size, in bytes, of the initial global arena. Defaults to half of
   * the available memory on the current device.
   * @param maximum_size Maximum size, in bytes, that the global arena can grow to. Defaults to all
   * of the available memory on the current device.
   */
  global_arena(Upstream* upstream_mr, std::size_t initial_size, std::size_t maximum_size)
    : upstream_mr_{upstream_mr}, maximum_size_{maximum_size}
  {
    RMM_EXPECTS(nullptr != upstream_mr_, "Unexpected null upstream pointer.");
    RMM_EXPECTS(initial_size == default_initial_size || initial_size == align_up(initial_size),
                "Error, Initial arena size required to be a multiple of 256 bytes");
    RMM_EXPECTS(maximum_size_ == default_maximum_size || maximum_size_ == align_up(maximum_size_),
                "Error, Maximum arena size required to be a multiple of 256 bytes");

    if (initial_size == default_initial_size || maximum_size == default_maximum_size) {
      std::size_t free{};
      std::size_t total{};
      RMM_CUDA_TRY(cudaMemGetInfo(&free, &total));
      if (initial_size == default_initial_size) {
        initial_size = align_up(std::min(free, total / 2));
      }
      if (maximum_size_ == default_maximum_size) {
        maximum_size_ = align_down(free) - reserved_size;
      }
    }
    RMM_EXPECTS(initial_size <= maximum_size_, "Initial arena size exceeds the maximum pool size!");

    free_blocks_.emplace(expand_arena(initial_size));
  }

  // Disable copy (and move) semantics.
  global_arena(global_arena const&) = delete;
  global_arena& operator=(global_arena const&) = delete;
  global_arena(global_arena&&) noexcept        = delete;
  global_arena& operator=(global_arena&&) noexcept = delete;

  /**
   * @brief Destroy the global arena and deallocate all memory it allocated using the upstream
   * resource.
   */
  ~global_arena()
  {
    lock_guard lock(mtx_);
    for (auto const& blk : upstream_blocks_) {
      upstream_mr_->deallocate(blk.pointer(), blk.size());
    }
  }

  /**
   * @brief Allocates memory of size at least `bytes`.
   *
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled.
   *
   * @param bytes The size in bytes of the allocation.
   * @return void* Pointer to the newly allocated memory.
   */
  block allocate(std::size_t bytes)
  {
    lock_guard lock(mtx_);
    return get_block(bytes);
  }

  /**
   * @brief Deallocate memory pointed to by `blk`.
   *
   * @param blk Block to be deallocated.
   */
  void deallocate(block const& blk)
  {
    lock_guard lock(mtx_);
    coalesce_block(free_blocks_, blk);
  }

  /**
   * @brief Deallocate a set of free blocks from a dying arena.
   *
   * @param free_blocks The set of free blocks.
   */
  void deallocate(std::set<block> const& free_blocks)
  {
    lock_guard lock(mtx_);
    for (auto const& blk : free_blocks) {
      coalesce_block(free_blocks_, blk);
    }
  }

  /**
   * @brief Dump memory to log.
   *
   * @param logger the spdlog logger to use
   */
  void dump_memory_log(std::shared_ptr<spdlog::logger> const& logger) const
  {
    lock_guard lock(mtx_);

    logger->info("  Maximum size: {}", rmm::detail::bytes{maximum_size_});
    logger->info("  Current size: {}", rmm::detail::bytes{current_size_});

    logger->info("  # free blocks: {}", free_blocks_.size());
    if (!free_blocks_.empty()) {
      logger->info("  Total size of free blocks: {}",
                   rmm::detail::bytes{total_block_size(free_blocks_)});
      auto const largest_free =
        *std::max_element(free_blocks_.begin(), free_blocks_.end(), block_size_compare);
      logger->info("  Size of largest free block: {}", rmm::detail::bytes{largest_free.size()});
    }

    logger->info("  # upstream blocks={}", upstream_blocks_.size());
    logger->info("  Total size of upstream blocks: {}",
                 rmm::detail::bytes{total_block_size(upstream_blocks_)});
  }

 private:
  using lock_guard = std::lock_guard<std::mutex>;

  /**
   * @brief Get an available memory block of at least `size` bytes.
   *
   * @param size The number of bytes to allocate.
   * @return block A block of memory of at least `size` bytes.
   */
  block get_block(std::size_t size)
  {
    // Find the first-fit free block.
    auto const blk = first_fit(free_blocks_, size);
    if (blk.is_valid()) { return blk; }

    // No existing larger blocks available, so grow the arena.
    auto const upstream_block = expand_arena(size_to_grow(size));
    coalesce_block(free_blocks_, upstream_block);
    return first_fit(free_blocks_, size);
  }

  /**
   * @brief Get the size to grow the global arena given the requested `size` bytes.
   *
   * This simply grows the global arena to the maximum size.
   *
   * @param size The number of bytes required.
   * @return size The size for the arena to grow, or 0 if no more memory.
   */
  constexpr std::size_t size_to_grow(std::size_t size) const
  {
    if (current_size_ + size > maximum_size_) { return 0; }
    return maximum_size_ - current_size_;
  }

  /**
   * @brief Allocate space from upstream to supply the arena and return a sufficiently sized block.
   *
   * @param size The minimum size to allocate.
   * @return block A block of at least `size` bytes.
   */
  block expand_arena(std::size_t size)
  {
    if (size > 0) {
      upstream_blocks_.push_back({upstream_mr_->allocate(size), size});
      current_size_ += size;
      return upstream_blocks_.back();
    }
    return {};
  }

  /// The upstream resource to allocate memory from.
  Upstream* upstream_mr_;
  /// The maximum size the global arena can grow to.
  std::size_t maximum_size_;
  /// The current size of the global arena.
  std::size_t current_size_{};
  /// Address-ordered set of free blocks.
  std::set<block> free_blocks_;
  /// Blocks allocated from upstream so that they can be quickly freed.
  std::vector<block> upstream_blocks_;
  /// Mutex for exclusive lock.
  mutable std::mutex mtx_;
};

/**
 * @brief An arena for allocating memory for a thread.
 *
 * An arena is a per-thread or per-non-default-stream memory pool. It allocates
 * superblocks from the global arena, and return them when the superblocks become empty.
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

  ~arena() = default;

  // Disable copy (and move) semantics.
  arena(arena const&) = delete;
  arena& operator=(arena const&) = delete;
  arena(arena&&) noexcept        = delete;
  arena& operator=(arena&&) noexcept = delete;

  /**
   * @brief Allocates memory of size at least `bytes`.
   *
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled.
   *
   * @param bytes The size in bytes of the allocation.
   * @return void* Pointer to the newly allocated memory.
   */
  void* allocate(std::size_t bytes)
  {
    lock_guard lock(mtx_);
    auto const blk = get_block(bytes);
    return blk.pointer();
  }

  /**
   * @brief Deallocate memory pointed to by `ptr`, and possibly return superblocks to upstream.
   *
   * @param ptr Pointer to be deallocated.
   * @param bytes The size in bytes of the allocation. This must be equal to the value of `bytes`
   * that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation.
   */
  void deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream)
  {
    lock_guard lock(mtx_);
    block const blk{ptr, bytes};
    auto const merged = coalesce_block(free_blocks_, blk);
    shrink_arena(merged, stream);
  }

  /**
   * @brief Clean the arena and deallocate free blocks from the global arena.
   */
  void clean()
  {
    lock_guard lock(mtx_);
    global_arena_.deallocate(free_blocks_);
    free_blocks_.clear();
  }

  /**
   * Dump memory to log.
   *
   * @param logger the spdlog logger to use
   */
  void dump_memory_log(std::shared_ptr<spdlog::logger> const& logger) const
  {
    lock_guard lock(mtx_);
    logger->info("    # free blocks: {}", free_blocks_.size());
    if (!free_blocks_.empty()) {
      logger->info("    Total size of free blocks: {}",
                   rmm::detail::bytes{total_block_size(free_blocks_)});
      auto const largest_free =
        *std::max_element(free_blocks_.begin(), free_blocks_.end(), block_size_compare);
      logger->info("    Size of largest free block: {}", rmm::detail::bytes{largest_free.size()});
    }
  }

 private:
  using lock_guard = std::lock_guard<std::mutex>;
  /// Maximum number of free blocks to keep.
  static constexpr int max_free_blocks = 16;

  /**
   * @brief Get an available memory block of at least `size` bytes.
   *
   * @param size The number of bytes to allocate.
   * @return A block of memory of at least `size` bytes.
   */
  block get_block(std::size_t size)
  {
    if (size < minimum_superblock_size) {
      // Find the first-fit free block.
      auto const blk = first_fit(free_blocks_, size);
      if (blk.is_valid()) { return blk; }
    }

    // No existing larger blocks available, so grow the arena and obtain a superblock.
    auto const superblock = expand_arena(size);
    if (superblock.is_valid()) {
      coalesce_block(free_blocks_, superblock);
      return first_fit(free_blocks_, size);
    }
    return superblock;
  }

  /**
   * @brief Allocate space from upstream to supply the arena and return a superblock.
   *
   * @return A superblock.
   */
  block expand_arena(std::size_t size)
  {
    auto const superblock_size = std::max(size, minimum_superblock_size);
    return global_arena_.allocate(superblock_size);
  }

  /**
   * @brief Shrink this arena by returning free superblocks to upstream.
   *
   * @param blk The block that can be used to shrink the arena.
   * @param stream Stream on which to perform shrinking.
   */
  void shrink_arena(block const& blk, cuda_stream_view stream)
  {
    if (blk.is_superblock() || free_blocks_.size() > max_free_blocks) {
      stream.synchronize_no_throw();
      global_arena_.deallocate(blk);
      free_blocks_.erase(blk);
    }
  }

  /// The global arena to allocate superblocks from.
  global_arena<Upstream>& global_arena_;
  /// Free blocks.
  std::set<block> free_blocks_;
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
  arena_cleaner(arena_cleaner const&) = delete;
  arena_cleaner& operator=(arena_cleaner const&) = delete;
  arena_cleaner(arena_cleaner&&) noexcept        = delete;
  arena_cleaner& operator=(arena_cleaner&&) = delete;

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
