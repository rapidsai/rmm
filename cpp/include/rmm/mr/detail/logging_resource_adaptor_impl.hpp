/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/logger.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <memory>
#include <string>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

/**
 * @brief Implementation class for logging_resource_adaptor.
 *
 * This class satisfies the CCCL `cuda::mr::resource` concept and provides
 * the actual logging functionality. It is held by `logging_resource_adaptor`
 * via `cuda::mr::shared_resource` for reference-counted ownership.
 */
class logging_resource_adaptor_impl {
 public:
  logging_resource_adaptor_impl(std::shared_ptr<rapids_logger::logger> logger,
                                device_async_resource_ref upstream,
                                bool auto_flush);

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t));

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept;

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t));

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept;

  bool operator==(logging_resource_adaptor_impl const& other) const noexcept
  {
    return upstream_ == other.upstream_ && logger_ == other.logger_;
  }

  bool operator!=(logging_resource_adaptor_impl const& other) const noexcept
  {
    return !(*this == other);
  }

  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept;

  void flush();

  [[nodiscard]] std::string header() const;

  RMM_CONSTEXPR_FRIEND void get_property(logging_resource_adaptor_impl const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 private:
  std::shared_ptr<rapids_logger::logger> logger_{};
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
};

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
