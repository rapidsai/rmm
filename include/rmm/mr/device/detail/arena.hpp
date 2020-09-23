/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <rmm/detail/aligned.hpp>
#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>

namespace rmm {
namespace mr {
namespace detail {
namespace arena {

constexpr std::size_t superblock_size = 1u << 26u;  ///< Size of a superblock (64 MiB).

/**
 * @brief Represents a chunk of memory that can be allocated and deallocated.
 *
 * A fixed-sized block obtained from the global arena is called a "superblock". Only a superblock
 * can be returned to the global arena.
 */
struct block {
  void* pointer{};     ///< Raw memory pointer.
  std::size_t size{};  ///< Size in bytes.
  bool is_head{};      ///< Indicates whether pointer was allocated from upstream.

  /// Returns true if this block is valid (non-null), false otherwise.
  bool is_valid() const { return pointer != nullptr; }

  /// Returns true if this block is a superblock, false otherwise.
  bool is_superblock() const { return is_head && size == superblock_size; }

  /**
   * @brief Verifies whether this block can be merged to the beginning of block b.
   *
   * @param b The block to check for contiguity.
   * @return true Returns true if this block's `pointer` + `size` == `b.ptr`, and `not b.is_head`,
                  false otherwise.
   */
  bool is_contiguous_before(block const& b) const
  {
    return increment(pointer, size) == b.pointer and not(b.is_head);
  }

  /**
   * @brief Is this block large enough to fit `sz` bytes?
   *
   * @param sz The size in bytes to check for fit.
   * @return true if this block is at least `sz` bytes.
   */
  bool fits(std::size_t sz) const { return size >= sz; }

  /**
   * @brief Split this block into two by the given size.
   *
   * @param sz The size in bytes of the first block.
   * @return std::pair<block, block> A pair of blocks split by sz.
   */
  std::pair<block, block> split(std::size_t sz) const
  {
    RMM_LOGGING_ASSERT(size >= sz);
    if (size > sz) {
      return {{pointer, sz, is_head}, {increment(pointer, sz), size - sz, false}};
    } else {
      return {*this, {}};
    }
  }

  /**
   * @brief Coalesce two contiguous blocks into one.
   *
   * `this` must immediately precede `b` and both `this` and `b` must be from the same superblock.
   *
   * @param b block to merge.
   * @return block The merged block.
   */
  block merge(block const& b) const
  {
    RMM_LOGGING_ASSERT(is_contiguous_before(b));
    return {pointer, size + b.size, is_head};
  }

  /**
   * @brief Increment the given pointer by the given size.
   *
   * @param pointer the pointer to increment.
   * @param size the size to add to the pointer.
   * @return void* The resulting pointer.
   */
  static void* increment(void* pointer, std::size_t size)
  {
    return static_cast<void*>(static_cast<char*>(pointer) + size);
  }

  /// Used by std::set to compare blocks.
  bool operator<(block const& b) const { return pointer < b.pointer; }
};

/// The required allocation alignment.
constexpr std::size_t allocation_alignment = 256;

/**
 * @brief Align up to the allocation alignment.
 *
 * @param[in] v value to align
 * @return Return the aligned value
 */
constexpr std::size_t align_up(std::size_t v) noexcept
{
  return rmm::detail::align_up(v, allocation_alignment);
}

/**
 * @brief Align down to the allocation alignment.
 *
 * @param[in] v value to align
 * @return Return the aligned value
 */
constexpr std::size_t align_down(std::size_t v) noexcept
{
  return rmm::detail::align_down(v, allocation_alignment);
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
    free_blocks.cbegin(), free_blocks.cend(), [size](auto const& b) { return b.fits(size); });

  if (iter == free_blocks.cend()) {
    return {};
  } else {
    // Remove the block from the free_list.
    auto const b = *iter;
    auto const i = free_blocks.erase(iter);

    // Split the block and put the remainder back.
    auto const split     = b.split(size);
    auto const remainder = split.second;
    if (remainder.is_valid()) { free_blocks.insert(i, remainder); }
    return split.first;
  }
}

/**
 * @brief Splits block `b` if necessary to return the block allocated.
 *
 * If the block is split, the remainder is returned to the free blocks.
 *
 * @param free_blocks The address-ordered set of free blocks.
 * @param b The block to allocate from.
 * @param size The size in bytes of the requested allocation.
 * @return block The allocated block.
 */
inline block split_block(std::set<block>& free_blocks, block const& b, std::size_t size)
{
  auto const split     = b.split(size);
  auto const remainder = split.second;
  if (remainder.is_valid()) free_blocks.emplace(remainder);
  return split.first;
}

/**
 * @brief Coalesce the given block with other free blocks.
 *
 * @param free_blocks The address-ordered set of free blocks.
 * @param b The block to coalesce.
 * @return block The coalesced block.
 */
inline block coalesce_block(std::set<block>& free_blocks, block const& b)
{
  if (!b.is_valid()) return b;

  // Find the right place (in ascending address order) to insert the block.
  auto const next     = free_blocks.lower_bound(b);
  auto const previous = next == free_blocks.cbegin() ? next : std::prev(next);

  // Coalesce with neighboring blocks.
  bool const merge_prev = previous->is_contiguous_before(b);
  bool const merge_next = next != free_blocks.cend() && b.is_contiguous_before(*next);

  block merged{};
  if (merge_prev && merge_next) {
    merged = previous->merge(b).merge(*next);
    free_blocks.erase(previous);
    auto const i = free_blocks.erase(next);
    free_blocks.insert(i, merged);
  } else if (merge_prev) {
    merged       = previous->merge(b);
    auto const i = free_blocks.erase(previous);
    free_blocks.insert(i, merged);
  } else if (merge_next) {
    merged       = b.merge(*next);
    auto const i = free_blocks.erase(next);
    free_blocks.insert(i, merged);
  } else {
    free_blocks.emplace(b);
    merged = b;
  }
  return merged;
}

/**
 * @brief The global arena for allocating memory from the upstream memory resource.
 *
 * The global arena is a shared memory pool from which other arenas allocate superblocks. Allocation
 * requests with size bigger than half of the size of a superblock is also handled directly by the
 * global arena.
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
  static constexpr std::size_t reserved_size = 1u << 26u;

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

    std::size_t free{}, total{};
    RMM_CUDA_TRY(cudaMemGetInfo(&free, &total));
    if (initial_size == default_initial_size) {
      initial_size = align_up(std::min(free, total / 2));
    }
    if (maximum_size_ == default_maximum_size) { maximum_size_ = align_down(free) - reserved_size; }

    RMM_EXPECTS(initial_size <= maximum_size_, "Initial arena size exceeds the maximum pool size!");

    free_blocks_.emplace(expand_arena(initial_size));
  }

  // Disable copy (and move) semantics.
  global_arena(const global_arena&) = delete;
  global_arena& operator=(const global_arena&) = delete;

  /**
   * @brief Destroy the global arena and deallocate all memory it allocated using the upstream
   * resource.
   */
  ~global_arena()
  {
    lock_guard lock(mtx_);
    for (auto const& b : upstream_blocks_) {
      upstream_mr_->deallocate(b.pointer, b.size);
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
  void* allocate(std::size_t bytes)
  {
    lock_guard lock(mtx_);
    auto const b = get_block(bytes);
    allocated_blocks_.emplace(b.pointer, b);
    return b.pointer;
  }

  /**
   * @brief Deallocate memory pointed to by `p`.
   *
   * @param p Pointer to be deallocated.
   * @param bytes The size in bytes of the allocation. This must be equal to the value of `bytes`
   * that was passed to the `allocate` call that returned `p`.
   */
  void deallocate(void* p, std::size_t bytes)
  {
    lock_guard lock(mtx_);
    auto const b = free_block(p, bytes);
    coalesce_block(free_blocks_, b);
  }

  /**
   * @brief Transfer the free and allocated blocks from a dying arena here.
   *
   * @param free_blocks The set of free blocks.
   * @param allocated_blocks The map of pointer address to allocated blocks.
   */
  void transfer(std::set<block> const& free_blocks,
                std::unordered_map<void*, block> const& allocated_blocks)
  {
    lock_guard lock(mtx_);
    for (auto const& b : free_blocks) {
      coalesce_block(free_blocks_, b);
    }
    for (auto const& kv : allocated_blocks) {
      allocated_blocks_.emplace(kv);
    }
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
    auto const b = first_fit(free_blocks_, size);
    if (b.is_valid()) return b;

    // No existing larger blocks available, so grow the arena.
    auto const upstream = expand_arena(size_to_grow(size));
    return split_block(free_blocks_, upstream, size);
  }

  /**
   * @brief Get the size to grow the global arena given the requested `size` bytes.
   *
   * This simply grows the global arena to the maximum size.
   *
   * @param size The number of bytes required.
   * @return size The size for the arena to grow.
   */
  constexpr std::size_t size_to_grow(std::size_t size) const
  {
    if (current_size_ + size > maximum_size_) {
      RMM_FAIL("Maximum pool size exceeded", rmm::bad_alloc);
    }
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
    upstream_blocks_.push_back({upstream_mr_->allocate(size), size, true});
    current_size_ += size;
    return upstream_blocks_.back();
  }

  /**
   * @brief Finds, frees and returns the block associated with pointer `p`.
   *
   * @param p The pointer to the memory to free.
   * @param size The size of the memory to free. Must be equal to the original allocation size.
   * @return The (now freed) block associated with `p`. The caller is expected to return the block
   * to the arena.
   */
  block free_block(void* p, std::size_t size) noexcept
  {
    auto const i = allocated_blocks_.find(p);
    RMM_LOGGING_ASSERT(i != allocated_blocks_.end());

    auto found = i->second;
    RMM_LOGGING_ASSERT(found.size == size);
    allocated_blocks_.erase(i);

    return found;
  }

  /// The upstream resource to allocate memory from.
  Upstream* upstream_mr_;
  /// The maximum size the global arena can grow to.
  std::size_t maximum_size_;
  /// The current size of the global arena.
  std::size_t current_size_{};
  /// Address-ordered set of free blocks.
  std::set<block> free_blocks_;
  /// Map of pointer address to allocated blocks.
  std::unordered_map<void*, block> allocated_blocks_;
  /// Blocks allocated from upstream so that they can be quickly freed.
  std::vector<block> upstream_blocks_;
  /// Mutex for exclusive lock.
  mutable std::mutex mtx_;
};

/**
 * @brief An arena for allocating memory for a thread.
 *
 * An arena is a per-thread or per-non-default-stream memory pool for handling small-size
 * allocations. It allocates superblocks from the global arena, and return them when the superblocks
 * become empty.
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
  arena(const arena&) = delete;
  arena& operator=(const arena&) = delete;

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
    auto const b = get_block(bytes);
    allocated_blocks_.emplace(b.pointer, b);
    return b.pointer;
  }

  /**
   * @brief Deallocate memory pointed to by `p`, and possibly return superblocks to upstream.
   *
   * @param p Pointer to be deallocated.
   * @param bytes The size in bytes of the allocation. This must be equal to the value of `bytes`
   * that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation.
   * @return true if the allocation is found, false otherwise.
   */
  bool deallocate(void* p, std::size_t bytes, cudaStream_t stream)
  {
    lock_guard lock(mtx_);
    bool found = do_deallocate(p, bytes);
    shrink_arena(stream);
    return found;
  }

  /**
   * @brief Deallocate memory pointed to by `p`, keeping all free superblocks.
   *
   * This is done when deallocating from another arena. Since we don't have access to the CUDA
   * stream associated with this arena, it's not safe to return superblocks.
   *
   * @param p Pointer to be deallocated.
   * @param bytes The size in bytes of the allocation. This must be equal to the value of `bytes`
   * that was passed to the `allocate` call that returned `p`.
   * @return true if the allocation is found, false otherwise.
   */
  bool deallocate(void* p, std::size_t bytes)
  {
    lock_guard lock(mtx_);
    return do_deallocate(p, bytes);
  }

  /**
   * @brief Clean the arena and transfer free/allocated blocks to the global arena.
   *
   * This is only needed when a per-thread arena is about to die.
   */
  void clean()
  {
    lock_guard lock(mtx_);
    global_arena_.transfer(free_blocks_, allocated_blocks_);
    free_blocks_.clear();
    free_superblocks_.clear();
    allocated_blocks_.clear();
  }

  /**
   * @brief Does an arena handle blocks of this size?
   *
   * @param size The size in bytes to check.
   * @return true if blocks of this size are handled by an arena.
   */
  static bool handles_size(std::size_t size) { return size <= maximum_allocation_size; }

 private:
  /// The maximum allocation size handled by arenas.
  static constexpr std::size_t maximum_allocation_size = superblock_size / 2;

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
    auto const b = first_fit(free_blocks_, size);
    if (b.is_valid()) {
      free_superblocks_.erase(b.pointer);
      return b;
    }

    // No existing larger blocks available, so grow the arena and obtain a superblock.
    auto const superblock = expand_arena();
    return split_block(free_blocks_, superblock, size);
  }

  /**
   * @brief Allocate space from upstream to supply the arena and return a superblock.
   *
   * @return block A superblock.
   */
  block expand_arena() const
  {
    return {global_arena_.allocate(superblock_size), superblock_size, true};
  }

  /**
   * @brief Deallocate memory pointed to by `p`.
   *
   * @param p Pointer to be deallocated.
   * @param bytes The size in bytes of the allocation. This must be equal to the value of `bytes`
   * that was passed to the `allocate` call that returned `p`.
   * @return bool if the allocation is found.
   */
  bool do_deallocate(void* p, std::size_t bytes)
  {
    auto const b      = free_block(p, bytes);
    auto const merged = coalesce_block(free_blocks_, b);
    if (merged.is_superblock()) { free_superblocks_.insert(merged.pointer); }
    return b.is_valid();
  }

  /**
   * @brief Finds, frees and returns the block associated with pointer `p`.
   *
   * @param p The pointer to the memory to free.
   * @param size The size of the memory to free. Must be equal to the original allocation size.
   * @return The (now freed) block associated with `p`. The caller is expected to return the block
   * to the arena.
   */
  block free_block(void* p, std::size_t size) noexcept
  {
    auto const i = allocated_blocks_.find(p);

    // The pointer may be allocated in another arena.
    if (i == allocated_blocks_.end()) { return {}; }

    auto const found = i->second;
    RMM_LOGGING_ASSERT(found.size == size);
    allocated_blocks_.erase(i);

    return found;
  }

  /**
   * @brief Shrink this arena by returning free superblocks to upstream.
   *
   * @param stream Stream on which to perform shrinking.
   */
  void shrink_arena(cudaStream_t stream)
  {
    // Don't shrink if only one free superblock is left (to avoid thrashing).
    if (free_superblocks_.size() <= 1) return;

    RMM_CUDA_TRY(cudaStreamSynchronize(stream));

    // Keep one superblock.
    for (auto it = std::next(free_superblocks_.cbegin()); it != free_superblocks_.cend(); ++it) {
      auto const pointer = *it;
      global_arena_.deallocate(pointer, superblock_size);
      free_superblocks_.erase(it--);
      free_blocks_.erase({pointer});
    }
  }

  /// The global arena to allocate superblocks from.
  global_arena<Upstream>& global_arena_;
  /// Free blocks including superblocks.
  std::set<block> free_blocks_;
  /// Free superblocks; these are also tracked in `free_blocks_`.
  std::set<void*> free_superblocks_;
  //// Map of pointer address to allocated blocks.
  std::unordered_map<void*, block> allocated_blocks_;
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
  explicit arena_cleaner(std::shared_ptr<arena<Upstream>> const& a) : arena_(a) {}

  // Disable copy (and move) semantics.
  arena_cleaner(const arena_cleaner&) = delete;
  arena_cleaner& operator=(const arena_cleaner&) = delete;

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

}  // namespace arena
}  // namespace detail
}  // namespace mr
}  // namespace rmm
