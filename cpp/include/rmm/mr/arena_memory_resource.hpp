/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/arena_memory_resource_impl.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief A suballocator that emphasizes fragmentation avoidance and scalable concurrency support.
 *
 * Allocation and deallocation are thread-safe. Also,
 * this class is compatible with CUDA per-thread default stream.
 *
 * GPU memory is divided into a global arena, per-thread arenas for default streams, and per-stream
 * arenas for non-default streams. Each arena allocates memory from the global arena in chunks
 * called superblocks.
 *
 * Blocks in each arena are allocated using address-ordered first fit. When a block is freed, it is
 * coalesced with neighbouring free blocks if the addresses are contiguous. Free superblocks are
 * returned to the global arena.
 *
 * In real-world applications, allocation sizes tend to follow a power law distribution in which
 * large allocations are rare, but small ones quite common. By handling small allocations in the
 * per-thread arena, adequate performance can be achieved without introducing excessive memory
 * fragmentation under high concurrency.
 *
 * This design is inspired by several existing CPU memory allocators targeting multi-threaded
 * applications (glibc malloc, Hoard, jemalloc, TCMalloc), albeit in a simpler form. Possible future
 * improvements include using size classes, allocation caches, and more fine-grained locking or
 * lock-free approaches.
 *
 * This class is copyable and shares ownership of its internal state via
 * `cuda::mr::shared_resource`.
 *
 * \see Wilson, P. R., Johnstone, M. S., Neely, M., & Boles, D. (1995, September). Dynamic storage
 * allocation: A survey and critical review. In International Workshop on Memory Management (pp.
 * 1-116). Springer, Berlin, Heidelberg.
 * \see Berger, E. D., McKinley, K. S., Blumofe, R. D., & Wilson, P. R. (2000). Hoard: A scalable
 * memory allocator for multithreaded applications. ACM Sigplan Notices, 35(11), 117-128.
 * \see Evans, J. (2006, April). A scalable concurrent malloc (3) implementation for FreeBSD. In
 * Proc. of the bsdcan conference, ottawa, canada.
 * \see https://sourceware.org/glibc/wiki/MallocInternals
 * \see http://hoard.org/
 * \see http://jemalloc.net/
 * \see https://github.com/google/tcmalloc
 */
class RMM_EXPORT arena_memory_resource
  : public cuda::mr::shared_resource<detail::arena_memory_resource_impl> {
  using shared_base = cuda::mr::shared_resource<detail::arena_memory_resource_impl>;

 public:
  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   */
  RMM_CONSTEXPR_FRIEND void get_property(arena_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Construct an `arena_memory_resource`.
   *
   * @tparam Upstream Type of the upstream resource (must be convertible to
   * `cuda::mr::any_resource<cuda::mr::device_accessible>`).
   * @param upstream_mr The memory resource from which to allocate blocks for the global arena.
   * @param arena_size Size in bytes of the global arena. Defaults to half of the available
   * memory on the current device.
   * @param dump_log_on_failure If true, dump memory log when running out of memory.
   */
  template <
    class Upstream,
    std::enable_if_t<!std::is_same_v<std::decay_t<Upstream>, arena_memory_resource>, int> = 0>
  explicit arena_memory_resource(Upstream&& upstream_mr,
                                 std::optional<std::size_t> arena_size = std::nullopt,
                                 bool dump_log_on_failure              = false)
    : shared_base(cuda::mr::make_shared_resource<detail::arena_memory_resource_impl>(
        cuda::mr::any_resource<cuda::mr::device_accessible>{std::forward<Upstream>(upstream_mr)},
        arena_size,
        dump_log_on_failure))
  {
  }

  ~arena_memory_resource() = default;
};

static_assert(cuda::mr::resource_with<arena_memory_resource, cuda::mr::device_accessible>,
              "arena_memory_resource does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
