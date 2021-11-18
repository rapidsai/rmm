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
#include <rmm/detail/cuda_util.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/logger.hpp>

#include <cuda_runtime_api.h>

#include <spdlog/common.h>
#include <spdlog/fmt/ostr.h>

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
 * @brief Represents a contiguous region of memory.
 */
class memory_span {
 public:
  /**
   * @brief Construct a default span.
   */
  memory_span() = default;

  /**
   * @brief Construct a span given a pointer and size.
   *
   * @param pointer The address for the beginning of the span.
   * @param size The size of the span.
   */
  memory_span(void* pointer, std::size_t size) : pointer_{static_cast<char*>(pointer)}, size_{size}
  {
    RMM_LOGGING_ASSERT(pointer != nullptr);
    RMM_LOGGING_ASSERT(size > 0);
  }

  /// Returns the underlying pointer.
  [[nodiscard]] char* pointer() const { return pointer_; }

  /// Returns the size of the span.
  [[nodiscard]] std::size_t size() const { return size_; }

  /// Returns true if this span is valid (non-null), false otherwise.
  [[nodiscard]] bool is_valid() const { return pointer_ != nullptr && size_ > 0; }

  /// Used by std::set to compare spans.
  bool operator<(memory_span const& ms) const
  {
    RMM_LOGGING_ASSERT(ms.is_valid());
    return pointer_ < ms.pointer_;
  }

 private:
  char* pointer_{};     ///< Raw memory pointer.
  std::size_t size_{};  ///< Size in bytes.
};

/// Calculate the total size of a set of memory spans.
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
class block final : public memory_span {
 public:
  using memory_span::memory_span;

  /**
   * @brief Is this block large enough to fit `sz` bytes?
   *
   * @param sz The size in bytes to check for fit.
   * @return true if this block is at least `sz` bytes.
   */
  [[nodiscard]] bool fits(std::size_t sz) const
  {
    RMM_LOGGING_ASSERT(is_valid());
    RMM_LOGGING_ASSERT(sz > 0);
    return size() >= sz;
  }

  /**
   * @brief Verifies whether this block can be merged to the beginning of block b.
   *
   * @param b The block to check for contiguity.
   * @return true Returns true if this block's `pointer` + `size` == `b.pointer`.
   */
  [[nodiscard]] bool is_contiguous_before(block const& b) const
  {
    RMM_LOGGING_ASSERT(is_valid());
    RMM_LOGGING_ASSERT(b.is_valid());
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return pointer() + size() == b.pointer();
  }

  /**
   * @brief Split this block into two by the given size.
   *
   * @param sz The size in bytes of the first block.
   * @return std::pair<block, block> A pair of blocks split by sz.
   */
  [[nodiscard]] std::pair<block, block> split(std::size_t sz) const
  {
    RMM_LOGGING_ASSERT(is_valid());
    RMM_LOGGING_ASSERT(size() >= sz);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return {{pointer(), sz}, {pointer() + sz, size() - sz}};
  }

  /**
   * @brief Coalesce two contiguous blocks into one.
   *
   * `this->is_contiguous_before(b)` must be true.
   *
   * @param b block to merge.
   * @return block The merged block.
   */
  [[nodiscard]] block merge(block const& b) const
  {
    RMM_LOGGING_ASSERT(is_valid());
    RMM_LOGGING_ASSERT(b.is_valid());
    RMM_LOGGING_ASSERT(is_contiguous_before(b));
    return {pointer(), size() + b.size()};
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
class superblock final : public memory_span {
 public:
  /// Minimum size of a superblock (4 MiB).
  static constexpr std::size_t minimum_size{1U << 22U};

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
  superblock(void* pointer, std::size_t size) : memory_span{pointer, size}
  {
    RMM_LOGGING_ASSERT(size > minimum_size / 2);
    free_blocks_.emplace(pointer, size);
  }

  // Disable copy semantics.
  superblock(superblock const&) = delete;
  superblock& operator=(superblock const&) = delete;
  // Allow move semantics.
  superblock(superblock&& sb) noexcept = default;
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
   * @brief Whether this superblock contains the given block.
   *
   * @param b The block to search for.
   * @return true if the given block belongs to this superblock.
   */
  [[nodiscard]] bool contains(block const& b) const
  {
    RMM_LOGGING_ASSERT(is_valid());
    RMM_LOGGING_ASSERT(b.is_valid());
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return pointer() <= b.pointer() && pointer() + size() >= b.pointer() + b.size();
  }

  /**
   * @brief Can this superblock fit `sz` bytes?
   *
   * @param sz The size in bytes to check for fit.
   * @return true if this superblock can fit `sz` bytes.
   */
  [[nodiscard]] bool fits(std::size_t sz) const
  {
    RMM_LOGGING_ASSERT(is_valid());
    return std::any_of(
      free_blocks_.cbegin(), free_blocks_.cend(), [sz](auto const& b) { return b.fits(sz); });
  }

  /**
   * @brief Verifies whether this superblock can be merged to the beginning of superblock s.
   *
   * @param s The superblock to check for contiguity.
   * @return true Returns true if both superblocks are empty and this superblock's
   * `pointer` + `size` == `s.ptr`.
   */
  [[nodiscard]] bool is_contiguous_before(superblock const& sb) const
  {
    RMM_LOGGING_ASSERT(is_valid());
    RMM_LOGGING_ASSERT(sb.is_valid());
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return empty() && sb.empty() && pointer() + size() == sb.pointer();
  }

  /**
   * @brief Split this superblock into two by the given size.
   *
   * @param sz The size in bytes of the first block.
   * @return superblock_pair A pair of superblocks split by sz.
   */
  [[nodiscard]] std::pair<superblock, superblock> split(std::size_t sz) const
  {
    RMM_LOGGING_ASSERT(is_valid());
    RMM_LOGGING_ASSERT(empty() && sz >= minimum_size && size() - sz >= minimum_size);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return {superblock{pointer(), sz}, superblock{pointer() + sz, size() - sz}};
  }

  /**
   * @brief Coalesce two contiguous superblocks into one.
   *
   * `this->is_contiguous_before(s)` must be true.
   *
   * @param sb superblock to merge.
   * @return block The merged block.
   */
  [[nodiscard]] superblock merge(superblock const& sb) const
  {
    RMM_LOGGING_ASSERT(is_contiguous_before(sb));
    return {pointer(), size() + sb.size()};
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

    auto const iter = std::find_if(
      free_blocks_.cbegin(), free_blocks_.cend(), [size](auto const& b) { return b.fits(size); });
    if (iter == free_blocks_.cend()) { return {}; }

    // Remove the block from the free list.
    auto const b    = *iter;
    auto const next = free_blocks_.erase(iter);

    if (b.size() > size) {
      // Split the block and put the remainder back.
      auto const split = b.split(size);
      free_blocks_.insert(next, split.second);
      return split.first;
    }
    return b;
  }

  /**
   * @brief Coalesce the given block with other free blocks.
   *
   * @param b The block to coalesce.
   */
  void coalesce(block const& b)
  {
    RMM_LOGGING_ASSERT(is_valid());
    RMM_LOGGING_ASSERT(b.is_valid());
    RMM_LOGGING_ASSERT(contains(b));

    if (free_blocks_.empty()) {
      free_blocks_.insert(b);
      return;
    }

    // Find the right place (in ascending address order) to insert the block.
    auto const next     = free_blocks_.lower_bound(b);
    auto const previous = next == free_blocks_.cbegin() ? next : std::prev(next);

    // Coalesce with neighboring blocks.
    bool const merge_prev = previous->is_contiguous_before(b);
    bool const merge_next = next != free_blocks_.cend() && b.is_contiguous_before(*next);

    if (merge_prev && merge_next) {
      auto const merged = previous->merge(b).merge(*next);
      free_blocks_.erase(previous);
      auto const iter = free_blocks_.erase(next);
      free_blocks_.insert(iter, merged);
    } else if (merge_prev) {
      auto const merged = previous->merge(b);
      auto const iter   = free_blocks_.erase(previous);
      free_blocks_.insert(iter, merged);
    } else if (merge_next) {
      auto const merged = b.merge(*next);
      auto const iter   = free_blocks_.erase(next);
      free_blocks_.insert(iter, merged);
    } else {
      free_blocks_.insert(next, b);
    }
  }

  /**
   * @brief Find the max free block size.
   * @return the max free block size.
   */
  [[nodiscard]] std::size_t max_free() const
  {
    return std::max_element(free_blocks_.cbegin(), free_blocks_.cend(), block_size_compare)->size();
  }

 private:
  /// Address-ordered set of free blocks.
  std::set<block> free_blocks_{};
};

/// Find the max free size from a set of superblocks.
inline auto max_free(std::set<superblock> const& superblocks)
{
  std::size_t size{};
  for (auto const& sb : superblocks) {
    size = std::max(size, sb.max_free());
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
    auto const size = rmm::detail::align_down(arena_size.value_or(default_size()),
                                              rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
    initialize(size);
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
    upstream_mr_->deallocate(upstream_block_.pointer(), upstream_block_.size());
    superblocks_.clear();
  }

  /**
   * @brief Should allocation of `size` bytes be handled by the global arena directly?
   *
   * @param size The size in bytes of the allocation.
   * @return bool True if the allocation should be handled by the global arena.
   */
  bool handles(std::size_t size) const { return size > superblock::minimum_size / 2; }

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
    lock_guard lock(mtx_);
    return first_fit(size);
  }

  /**
   * @brief Release a superblock.
   *
   * @param s Superblock to be released.
   * @param stream The stream to synchronize on before releasing.
   */
  void release(superblock&& sb, cuda_stream_view stream)
  {
    RMM_LOGGING_ASSERT(sb.is_valid());
    stream.synchronize_no_throw();
    lock_guard lock(mtx_);
    coalesce(std::move(sb));
  }

  /**
   * @brief Release a set of superblocks from a dying arena.
   *
   * @param superblocks The set of superblocks.
   */
  void release(std::set<superblock>& superblocks)
  {
    lock_guard lock(mtx_);
    while (!superblocks.empty()) {
      auto sb = std::move(superblocks.extract(superblocks.cbegin()).value());
      RMM_LOGGING_ASSERT(sb.is_valid());
      coalesce(std::move(sb));
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
    lock_guard lock(mtx_);
    return first_fit(size).pointer();
  }

  /**
   * @brief Deallocate memory pointed to by `ptr` directly.
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
    lock_guard lock(mtx_);
    coalesce({ptr, size});
    return true;
  }

  /**
   * @brief Deallocate memory pointed to by `ptr` that was allocated in a per-thread arena.
   *
   * @param ptr Pointer to be deallocated.
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `ptr`.
   * @param stream Stream on which to perform deallocation.
   */
  void deallocate_from_other_arena(void* ptr, std::size_t bytes)
  {
    lock_guard lock(mtx_);

    block const b{ptr, bytes};
    auto const iter = std::find_if(
      superblocks_.cbegin(), superblocks_.cend(), [&](auto const& sb) { return sb.contains(b); });
    if (iter == superblocks_.cend()) { RMM_FAIL("allocation not found"); }

    auto sb = std::move(superblocks_.extract(iter).value());
    sb.coalesce(b);
    if (sb.empty()) {
      coalesce(std::move(sb));
    } else {
      superblocks_.insert(std::move(sb));
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

    logger->info("  Arena size: {}", rmm::detail::bytes{upstream_block_.size()});
    logger->info("  # superblocks: {}", superblocks_.size());
    if (!superblocks_.empty()) {
      logger->info("  Total size of superblocks: {}",
                   rmm::detail::bytes{total_memory_size(superblocks_)});
      logger->info("  Size of largest free block: {}", rmm::detail::bytes{max_free(superblocks_)});
    }
  }

 private:
  using lock_guard = std::lock_guard<std::mutex>;

  /**
   * @brief Default size of the global arena if unspecified.
   * @return the default global arena size.
   */
  constexpr std::size_t default_size() const
  {
    auto const [free, total] = rmm::detail::available_device_memory();
    return free / 2;
  }

  /**
   * @brief Allocate space from upstream to initialize the arena.
   *
   * @param size The size to allocate.
   */
  void initialize(std::size_t size)
  {
    RMM_LOGGING_ASSERT(size >= superblock::minimum_size);
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
   * @return superblock A superblock that can fit at least `size` bytes, or empty if not found.
   */
  superblock first_fit(std::size_t size)
  {
    auto const iter = std::find_if(
      superblocks_.cbegin(), superblocks_.cend(), [size](auto const& sb) { return sb.fits(size); });
    if (iter == superblocks_.cend()) { return {}; }

    auto sb       = std::move(superblocks_.extract(iter).value());
    auto const sz = std::max(size, superblock::minimum_size);
    if (sb.empty() && sb.size() - sz >= superblock::minimum_size) {
      // Split the superblock and put the remainder back.
      auto [head, tail] = sb.split(sz);
      superblocks_.insert(std::move(tail));
      return std::move(head);
    }
    return sb;
  }

  /**
   * @brief Coalesce the given superblock with other empty superblocks.
   *
   * @param sb The superblock to coalesce.
   */
  void coalesce(superblock&& sb)
  {
    RMM_LOGGING_ASSERT(sb.is_valid());

    if (superblocks_.empty()) {
      superblocks_.insert(std::move(sb));
      return;
    }

    // Find the right place (in ascending address order) to insert the block.
    auto const next     = superblocks_.lower_bound(sb);
    auto const previous = next == superblocks_.cbegin() ? next : std::prev(next);

    // Coalesce with neighboring blocks.
    bool const merge_prev = previous->is_contiguous_before(sb);
    bool const merge_next = next != superblocks_.cend() && sb.is_contiguous_before(*next);

    if (merge_prev && merge_next) {
      auto prev_sb = std::move(superblocks_.extract(previous).value());
      auto next_sb = std::move(superblocks_.extract(next).value());
      auto merged  = prev_sb.merge(sb).merge(next_sb);
      superblocks_.insert(std::move(merged));
    } else if (merge_prev) {
      auto prev_sb = std::move(superblocks_.extract(previous).value());
      auto merged  = prev_sb.merge(sb);
      superblocks_.insert(std::move(merged));
    } else if (merge_next) {
      auto next_sb = std::move(superblocks_.extract(next).value());
      auto merged  = sb.merge(next_sb);
      superblocks_.insert(std::move(merged));
    } else {
      superblocks_.insert(std::move(sb));
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
  arena(arena const&) = delete;
  arena& operator=(arena const&) = delete;
  arena(arena&&) noexcept        = delete;
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
    lock_guard lock(mtx_);
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
    if (global_arena_.handles(size)) { return global_arena_.deallocate(ptr, size, stream); }
    lock_guard lock(mtx_);
    return deallocate_from_superblock({ptr, size}, stream);
  }

  /**
   * @brief Clean the arena and deallocate free blocks from the global arena.
   */
  void clean()
  {
    lock_guard lock(mtx_);
    global_arena_.release(superblocks_);
  }

  /**
   * Dump memory to log.
   *
   * @param logger the spdlog logger to use
   */
  void dump_memory_log(std::shared_ptr<spdlog::logger> const& logger) const
  {
    lock_guard lock(mtx_);
    logger->info("    # superblocks: {}", superblocks_.size());
    if (!superblocks_.empty()) {
      logger->info("    Total size of superblocks: {}",
                   rmm::detail::bytes{total_memory_size(superblocks_)});
      logger->info("    Size of largest free block: {}",
                   rmm::detail::bytes{max_free(superblocks_)});
    }
  }

 private:
  using lock_guard = std::lock_guard<std::mutex>;

  /**
   * @brief Get an available memory block of at least `size` bytes.
   *
   * @param size The number of bytes to allocate.
   * @return A block of memory of at least `size` bytes.
   */
  block get_block(std::size_t size)
  {
    // Find the first-fit free block.
    auto const b = first_fit(size);
    if (b.is_valid()) { return b; }

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
    auto const iter = std::find_if(
      superblocks_.cbegin(), superblocks_.cend(), [size](auto const& sb) { return sb.fits(size); });
    if (iter == superblocks_.cend()) { return {}; }

    auto sb      = std::move(superblocks_.extract(iter).value());
    auto const b = sb.first_fit(size);
    superblocks_.insert(std::move(sb));
    return b;
  }

  /**
   * @brief Deallocate a block from the superblock it belongs to.
   *
   * @param b The block to deallocate.
   * @param stream The stream to use for deallocation.
   * @return true if the block is found.
   */
  bool deallocate_from_superblock(block const& b, cuda_stream_view stream)
  {
    auto const iter = std::find_if(
      superblocks_.cbegin(), superblocks_.cend(), [&](auto const& sb) { return sb.contains(b); });
    if (iter == superblocks_.cend()) { return false; }

    auto sb = std::move(superblocks_.extract(iter).value());
    sb.coalesce(b);
    if (sb.empty()) {
      global_arena_.release(std::move(sb), stream);
    } else {
      superblocks_.insert(std::move(sb));
    }
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
    auto sb = global_arena_.acquire(size);
    if (sb.is_valid()) {
      auto const b = sb.first_fit(size);
      superblocks_.insert(std::move(sb));
      return b;
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
