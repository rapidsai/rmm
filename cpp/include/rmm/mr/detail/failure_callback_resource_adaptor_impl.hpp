/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <functional>
#include <utility>

namespace RMM_NAMESPACE {
namespace mr {

/**
 * @brief Callback function type used by failure_callback_resource_adaptor
 *
 * The resource adaptor calls this function when a memory allocation throws a specified exception
 * type. The function decides whether the resource adaptor should try to allocate the memory again
 * or re-throw the exception.
 *
 * The callback function signature is:
 *     `bool failure_callback_t(std::size_t bytes, void* callback_arg)`
 *
 * The callback function is passed two parameters: `bytes` is the size of the failed memory
 * allocation and `arg` is the extra argument passed to the constructor of the
 * `failure_callback_resource_adaptor`. The callback function returns a Boolean where true means to
 * retry the memory allocation and false means to re-throw the exception.
 */
using failure_callback_t = std::function<bool(std::size_t, void*)>;

namespace detail {

/**
 * @brief Implementation class for failure_callback_resource_adaptor.
 *
 * @tparam ExceptionType The type of exception that this adaptor should respond to.
 */
template <typename ExceptionType>
class failure_callback_resource_adaptor_impl {
 public:
  failure_callback_resource_adaptor_impl(device_async_resource_ref upstream,
                                         failure_callback_t callback,
                                         void* callback_arg)
    : upstream_mr_{upstream}, callback_{std::move(callback)}, callback_arg_{callback_arg}
  {
  }

  ~failure_callback_resource_adaptor_impl() = default;

  failure_callback_resource_adaptor_impl(failure_callback_resource_adaptor_impl const&) = delete;
  failure_callback_resource_adaptor_impl(failure_callback_resource_adaptor_impl&&)      = delete;
  failure_callback_resource_adaptor_impl& operator=(failure_callback_resource_adaptor_impl const&) =
    delete;
  failure_callback_resource_adaptor_impl& operator=(failure_callback_resource_adaptor_impl&&) =
    delete;

  bool operator==(failure_callback_resource_adaptor_impl const& other) const noexcept
  {
    return this == std::addressof(other);
  }

  bool operator!=(failure_callback_resource_adaptor_impl const& other) const noexcept
  {
    return !(*this == other);
  }

  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept
  {
    return device_async_resource_ref{
      const_cast<cuda::mr::any_resource<cuda::mr::device_accessible>&>(upstream_mr_)};
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t /*alignment*/ = alignof(std::max_align_t))
  {
    void* ret{};
    while (true) {
      try {
        ret = upstream_mr_.allocate(stream, bytes);
        break;
      } catch (ExceptionType const&) {
        if (!callback_(bytes, callback_arg_)) { throw; }
      }
    }
    return ret;
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t /*alignment*/ = alignof(std::max_align_t)) noexcept
  {
    upstream_mr_.deallocate(stream, ptr, bytes);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    return allocate(cuda_stream_view{}, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    deallocate(cuda_stream_view{}, ptr, bytes, alignment);
  }

  RMM_CONSTEXPR_FRIEND void get_property(failure_callback_resource_adaptor_impl const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_mr_;
  failure_callback_t callback_;
  void* callback_arg_;
};

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
