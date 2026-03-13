/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <memory>
#include <shared_mutex>
#include <stack>
#include <utility>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

/**
 * @brief Implementation class for statistics_resource_adaptor.
 *
 * Tracks allocation statistics through the upstream resource. This class satisfies
 * the CCCL `cuda::mr::resource` concept and is held by `statistics_resource_adaptor`
 * via `cuda::mr::shared_resource` for reference-counted ownership.
 */
class statistics_resource_adaptor_impl {
 public:
  using read_lock_t  = std::shared_lock<std::shared_mutex>;
  using write_lock_t = std::unique_lock<std::shared_mutex>;

  struct counter {
    int64_t value{0};
    int64_t peak{0};
    int64_t total{0};

    counter& operator+=(int64_t val)
    {
      value += val;
      total += val;
      peak = std::max(value, peak);
      return *this;
    }

    counter& operator-=(int64_t val)
    {
      value -= val;
      return *this;
    }

    void add_counters_from_tracked_sub_block(counter const& val)
    {
      peak = std::max(value + val.peak, peak);
      value += val.value;
      total += val.total;
    }
  };

  explicit statistics_resource_adaptor_impl(device_async_resource_ref upstream);

  ~statistics_resource_adaptor_impl() = default;

  statistics_resource_adaptor_impl(statistics_resource_adaptor_impl const&)            = delete;
  statistics_resource_adaptor_impl(statistics_resource_adaptor_impl&&)                 = delete;
  statistics_resource_adaptor_impl& operator=(statistics_resource_adaptor_impl const&) = delete;
  statistics_resource_adaptor_impl& operator=(statistics_resource_adaptor_impl&&)      = delete;

  bool operator==(statistics_resource_adaptor_impl const& other) const noexcept
  {
    return this == std::addressof(other);
  }

  bool operator!=(statistics_resource_adaptor_impl const& other) const noexcept
  {
    return !(*this == other);
  }

  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  [[nodiscard]] counter get_bytes_counter() const noexcept;

  [[nodiscard]] counter get_allocations_counter() const noexcept;

  std::pair<counter, counter> push_counters();

  std::pair<counter, counter> pop_counters();

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t));

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept;

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t));

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept;

  RMM_CONSTEXPR_FRIEND void get_property(statistics_resource_adaptor_impl const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 private:
  // Stack of counter pairs <bytes, allocations>. Invariant: always >= 1 entry.
  std::stack<std::pair<counter, counter>> counter_stack_{{std::make_pair(counter{}, counter{})}};
  mutable std::shared_mutex mtx_;
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_mr_;
};

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
