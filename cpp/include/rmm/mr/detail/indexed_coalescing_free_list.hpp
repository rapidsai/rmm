/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/coalescing_free_list.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#ifdef RMM_DEBUG_PRINT
#include <iostream>
#endif
#include <iterator>
#include <map>
#include <set>

RMM_NAMESPACE_BEGIN
namespace mr::detail {
struct indexed_coalescing_free_list : free_list<block> {
 private:
  using base_type = free_list<block>;

  struct compare_iterator_sizes {
    using is_transparent = void;

    bool operator()(iterator const& lhs, iterator const& rhs) const noexcept
    {
      if (lhs->size() != rhs->size()) { return lhs->size() < rhs->size(); }
      return std::less<char*>{}(lhs->pointer(), rhs->pointer());
    }

    bool operator()(iterator const& lhs, std::size_t rhs) const noexcept
    { return lhs->size() < rhs; }

    bool operator()(std::size_t lhs, iterator const& rhs) const noexcept
    { return lhs < rhs->size(); }
  };

  using size_index    = std::set<iterator, compare_iterator_sizes>;
  using address_index = std::map<char*, iterator, std::less<>>;

  // Build the indexes only after fragmentation makes linear search expensive. Once enabled they
  // stay enabled for this free list's lifetime, avoiding transition churn and simplifying
  // invariants.
  static constexpr std::size_t enable_index_threshold{1024};

 public:
  indexed_coalescing_free_list()           = default;
  ~indexed_coalescing_free_list() override = default;

  indexed_coalescing_free_list(indexed_coalescing_free_list const&)            = delete;
  indexed_coalescing_free_list& operator=(indexed_coalescing_free_list const&) = delete;
  indexed_coalescing_free_list(indexed_coalescing_free_list&&)                 = delete;
  indexed_coalescing_free_list& operator=(indexed_coalescing_free_list&&)      = delete;

  /**
   * @brief Preallocated index nodes for a later allocation-free splice.
   */
  struct prepared_splice {
    typename size_index::node_type size_node{};
    typename address_index::node_type address_node{};
  };

  /**
   * @brief A best-fit block and its optional size-index entry.
   */
  struct block_selection {
    iterator block{};
    typename size_index::iterator size{};
  };

  /**
   * @brief Ensures a later splice of `additional_blocks` cannot trigger index construction.
   *
   * Index construction uses temporary containers and swaps only after every allocation succeeds.
   */
  void prepare_for_splice(std::size_t additional_blocks)
  {
    if (index_active_ || size() + additional_blocks < enable_index_threshold) { return; }

    size_index sizes;
    address_index addresses;
    for (auto iter = begin(); iter != end(); ++iter) {
      sizes.insert(iter);
      addresses.emplace(iter->pointer(), iter);
    }
    blocks_by_size_.swap(sizes);
    blocks_by_address_.swap(addresses);
    index_active_ = true;
  }

  /**
   * @brief Allocates index nodes for transferring `iter` into this list.
   *
   * Must be called before any external publication operation that makes rollback impossible.
   */
  [[nodiscard]] prepared_splice prepare_splice(iterator iter) const
  {
    prepared_splice result;
    if (!index_active_) { return result; }

    size_index sizes;
    address_index addresses;
    auto const size_iter    = sizes.insert(iter).first;
    auto const address_iter = addresses.emplace(iter->pointer(), iter).first;
    result.size_node        = sizes.extract(size_iter);
    result.address_node     = addresses.extract(address_iter);
    return result;
  }

  /**
   * @brief Extracts one exact node into `destination` without allocating.
   *
   * Destination is an unindexed staging list. The node remains stable and is no longer present in
   * this list's private indexes.
   */
  void extract_exact(iterator iter, indexed_coalescing_free_list& destination) noexcept
  {
    assert(!destination.index_active_);
    erase_from_indexes(iter);
    destination.base_type::splice(destination.cend(), static_cast<base_type&>(*this), iter);
  }

  /**
   * @brief Commits a pre-staged exact-node splice without allocation.
   */
  void commit_prepared_splice(indexed_coalescing_free_list& source,
                              iterator iter,
                              prepared_splice&& prepared) noexcept
  {
    assert(!source.index_active_);
    auto const next = [&]() -> iterator {
      if (!index_active_) {
        return std::find_if(
          begin(), end(), [iter](block_type const& candidate) { return *iter < candidate; });
      }
      auto const address_iter = blocks_by_address_.lower_bound(iter->pointer());
      return (address_iter == blocks_by_address_.end()) ? end() : address_iter->second;
    }();

    commit_prepared_splice(source, iter, next, std::move(prepared));
  }

  /**
   * @brief Commits a pre-staged exact-node splice at an already-resolved position.
   */
  void commit_prepared_splice(indexed_coalescing_free_list& source,
                              iterator iter,
                              iterator next,
                              prepared_splice&& prepared) noexcept
  {
    assert(!source.index_active_);
    assert(next == end() || std::less<char*>{}(iter->pointer(), next->pointer()));
    assert(next == begin() ||
           std::less<char*>{}(std::prev(next)->pointer(), iter->pointer()));

    base_type::splice(next, static_cast<base_type&>(source), iter);
    insert_index_nodes(std::move(prepared), iter);
    has_failed_lookup_ = false;
  }

  /**
   * @brief Inserts a block in pointer order and coalesces adjacent blocks.
   *
   * @param block The block to insert.
   */
  std::size_t insert(block_type const& block)
  {
    if (is_empty()) {
      insert_uncoalesced(block, end());
      return block.size();
    }

    auto const next = [&]() -> iterator {
      if (!index_active_) {
        return std::find_if(
          begin(), end(), [block](block_type const& candidate) { return block < candidate; });
      }
      auto const address_iter = blocks_by_address_.lower_bound(block.pointer());
      return (address_iter == blocks_by_address_.end()) ? end() : address_iter->second;
    }();
    auto const previous = (next == cbegin()) ? next : std::prev(next);

    bool const merge_prev = previous->is_contiguous_before(block);
    bool const merge_next = (next != cend()) && block.is_contiguous_before(*next);

    std::size_t inserted_size{};
    if (merge_prev && merge_next) {
      // Both entries must be detached before either indexed block is changed. The surviving
      // previous block's nodes can then be re-keyed and reinserted without allocating.
      auto previous_nodes                 = extract_index_nodes(previous);
      [[maybe_unused]] auto next_nodes     = extract_index_nodes(next);
      *previous                           = previous->merge(block).merge(*next);
      base_type::erase(next);
      insert_index_nodes(std::move(previous_nodes), previous);
      inserted_size = previous->size();
    } else if (merge_prev) {
      auto previous_nodes = extract_index_nodes(previous);
      *previous           = previous->merge(block);
      insert_index_nodes(std::move(previous_nodes), previous);
      inserted_size = previous->size();
    } else if (merge_next) {
      auto next_nodes = extract_index_nodes(next);
      *next           = block.merge(*next);
      insert_index_nodes(std::move(next_nodes), next);
      inserted_size = next->size();
    } else {
      insert_uncoalesced(block, next);
      inserted_size = block.size();
    }

    // Any committed insertion can create a fitting block directly or through coalescing.
    has_failed_lookup_ = false;
    return inserted_size;
  }

  /**
   * @brief Moves all blocks from `other` into this free list.
   *
   * @param other The free list whose blocks are inserted.
   */
  std::size_t insert(free_list&& other)
  {
    using std::make_move_iterator;
    std::size_t largest_inserted{};
    auto inserter = [this, &largest_inserted](block_type&& block) {
      largest_inserted = std::max(largest_inserted, this->insert(block));
    };
    std::for_each(make_move_iterator(other.begin()), make_move_iterator(other.end()), inserter);
    return largest_inserted;
  }

  /**
   * @brief Finds the smallest block at least `size` bytes without removing it.
   *
   * The returned iterator remains valid until this free list is modified.
   *
   * @param size Requested allocation size.
   * @return A best-fit selection whose block is `end()` if no block fits.
   */
  block_selection find_block(std::size_t size)
  {
    if (index_active_) {
      auto const size_iter = blocks_by_size_.lower_bound(size);
      return (size_iter == blocks_by_size_.cend()) ? block_selection{end(), {}}
                                                   : block_selection{*size_iter, size_iter};
    }

    // A failed lookup proves that this request and all larger requests cannot fit. Cache the
    // smallest such request so repeated failures are O(1), while successful small-list searches
    // keep the existing linear best-fit algorithm.
    if (has_failed_lookup_ && size >= smallest_known_failed_request_) {
      return block_selection{end(), {}};
    }

    auto finder = [size](block_type const& lhs, block_type const& rhs) {
      return lhs.is_better_fit(size, rhs);
    };
    auto const iter = std::min_element(begin(), end(), finder);
    if (iter != end() && iter->fits(size)) { return block_selection{iter, {}}; }

    smallest_known_failed_request_ =
      has_failed_lookup_ ? std::min(smallest_known_failed_request_, size) : size;
    has_failed_lookup_ = true;
    return block_selection{end(), {}};
  }

  /**
   * @brief Commits consumption of a selected block without allocating.
   *
   * If `remainder` is valid, it replaces the selected block in-place and reuses that block's
   * extracted index nodes. Otherwise the selected block is erased.
   *
   * @param selection Selection previously returned by `find_block`.
   * @param remainder Optional unallocated suffix of the selected block.
   */
  void commit_block_selection(block_selection selection, block_type const& remainder) noexcept
  {
    assert(selection.block != end());
    assert(!remainder.is_valid() ||
           (std::less<char*>{}(selection.block->pointer(), remainder.pointer()) &&
            remainder.size() < selection.block->size()));

    if (!remainder.is_valid()) {
      if (index_active_) {
        assert(selection.size != blocks_by_size_.end());
        assert(*selection.size == selection.block);
        auto const address = blocks_by_address_.find(selection.block->pointer());
        assert(address != blocks_by_address_.end());
        assert(address->second == selection.block);
        blocks_by_size_.erase(selection.size);
        blocks_by_address_.erase(address);
      }
      base_type::erase(selection.block);
      return;
    }

    auto nodes       = extract_index_nodes(selection);
    *selection.block = remainder;
    insert_index_nodes(std::move(nodes), selection.block);
  }

  /**
   * @brief Removes and returns a selected block.
   *
   * @param selection Selection previously returned by `find_block`.
   * @return The removed block.
   */
  block_type remove_block(block_selection selection) noexcept
  {
    assert(selection.block != end());
    block_type const found = *selection.block;
    commit_block_selection(selection, {});
    return found;
  }

  /**
   * @brief Removes and returns the smallest block at least `size` bytes.
   *
   * @param size Requested allocation size.
   * @return A best-fit block, or an invalid block if no block fits.
   */
  block_type get_block(std::size_t size)
  {
    auto const selection = find_block(size);
    return (selection.block == end()) ? block_type{} : remove_block(selection);
  }

  [[nodiscard]] bool diagnostics_index_active() const noexcept { return index_active_; }

  [[nodiscard]] bool diagnostics_indexes_consistent() const noexcept
  {
    if (!index_active_) { return blocks_by_size_.empty() && blocks_by_address_.empty(); }
    if (blocks_by_size_.size() != size() || blocks_by_address_.size() != size()) { return false; }
    for (auto const indexed_iter : blocks_by_size_) {
      auto const address_iter = blocks_by_address_.find(indexed_iter->pointer());
      if (address_iter == blocks_by_address_.end() || address_iter->second != indexed_iter) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Returns the size of the largest free block without changing the list.
   *
   * @return The largest block size, or zero when the list is empty.
   */
  [[nodiscard]] std::size_t largest_block_size() const noexcept
  {
    if (is_empty()) { return 0; }
    if (index_active_) { return (*blocks_by_size_.crbegin())->size(); }
    return std::max_element(
             cbegin(),
             cend(),
             [](auto const& lhs, auto const& rhs) { return lhs.size() < rhs.size(); })
      ->size();
  }

  void clear() noexcept
  {
    blocks_by_size_.clear();
    blocks_by_address_.clear();
    index_active_      = false;
    has_failed_lookup_ = false;
    base_type::clear();
  }

#ifdef RMM_DEBUG_PRINT
  void print() const
  {
    std::cout << size() << '\n';
    std::for_each(cbegin(), cend(), [](auto const iter) { iter.print(); });
  }
#endif

 private:
  /**
   * @brief Inserts a non-coalescing block after all required nodes have been allocated.
   */
  void insert_uncoalesced(block_type const& block, iterator next)
  {
    assert(next == end() || block < *next);
    assert(next == begin() || *std::prev(next) < block);

    // Below the activation threshold, std::list::insert itself has the strong guarantee.
    if (!index_active_ && size() + 1 < enable_index_threshold) {
      base_type::insert(next, block);
      has_failed_lookup_ = false;
      return;
    }

    // Stage the list node first so the iterator indexed below remains stable. Index activation does
    // not invalidate the already-resolved list position. Both live index entries are then allocated
    // before the no-throw splice commit; if address insertion fails, removing the size entry restores
    // the pre-insertion representation.
    indexed_coalescing_free_list staging;
    staging.base_type::insert(staging.cend(), block);
    auto const staged = staging.begin();
    prepare_for_splice(1);
    assert(index_active_);

    auto const [size_iter, size_inserted] = blocks_by_size_.insert(staged);
    if (!size_inserted) {
      RMM_FAIL("Duplicate block in indexed free-list size index");
    }

    auto const address_result = [&]() {
      try {
        return blocks_by_address_.emplace(staged->pointer(), staged);
      } catch (...) {
        blocks_by_size_.erase(size_iter);
        throw;
      }
    }();
    if (!address_result.second) {
      blocks_by_size_.erase(size_iter);
      RMM_FAIL("Duplicate block address in indexed free list");
    }

    base_type::splice(next, static_cast<base_type&>(staging), staged);
    has_failed_lookup_ = false;
  }

  /**
   * @brief Extracts both index nodes associated with `iter`.
   */
  [[nodiscard]] prepared_splice extract_index_nodes(iterator iter) noexcept
  {
    prepared_splice result;
    if (!index_active_) { return result; }

    result.size_node    = blocks_by_size_.extract(iter);
    result.address_node = blocks_by_address_.extract(iter->pointer());
    assert(!result.size_node.empty());
    assert(!result.address_node.empty());
    return result;
  }

  /**
   * @brief Extracts both index nodes using a previously resolved best-fit selection.
   */
  [[nodiscard]] prepared_splice extract_index_nodes(block_selection selection) noexcept
  {
    prepared_splice result;
    if (!index_active_) { return result; }

    assert(selection.size != blocks_by_size_.end());
    assert(*selection.size == selection.block);
    result.size_node = blocks_by_size_.extract(selection.size);

    auto const address = blocks_by_address_.find(selection.block->pointer());
    assert(address != blocks_by_address_.end());
    assert(address->second == selection.block);
    result.address_node = blocks_by_address_.extract(address);

    assert(!result.size_node.empty());
    assert(!result.address_node.empty());
    return result;
  }

  /**
   * @brief Reinserts prepared index nodes for `iter` without allocating.
   */
  void insert_index_nodes(prepared_splice&& prepared, iterator iter) noexcept
  {
    if (!index_active_) {
      assert(prepared.size_node.empty());
      assert(prepared.address_node.empty());
      return;
    }

    assert(!prepared.size_node.empty());
    assert(!prepared.address_node.empty());
    prepared.size_node.value()     = iter;
    prepared.address_node.key()    = iter->pointer();
    prepared.address_node.mapped() = iter;
    auto const size_result         = blocks_by_size_.insert(std::move(prepared.size_node));
    auto const address_result = blocks_by_address_.insert(std::move(prepared.address_node));
    assert(size_result.inserted);
    assert(address_result.inserted);
    (void)size_result;
    (void)address_result;
  }

  void erase_from_indexes(iterator iter)
  {
    if (!index_active_) { return; }
    auto const size_erased    = blocks_by_size_.erase(iter);
    auto const address_erased = blocks_by_address_.erase(iter->pointer());
    assert(size_erased == 1);
    assert(address_erased == 1);
    (void)size_erased;
    (void)address_erased;
  }

  size_index blocks_by_size_{};
  address_index blocks_by_address_{};
  bool index_active_{false};
  bool has_failed_lookup_{false};
  std::size_t smallest_known_failed_request_{};
};  // indexed_coalescing_free_list

}  // namespace mr::detail
RMM_NAMESPACE_END
