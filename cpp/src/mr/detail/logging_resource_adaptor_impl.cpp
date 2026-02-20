/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/format.hpp>
#include <rmm/mr/detail/logging_resource_adaptor_impl.hpp>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

logging_resource_adaptor_impl::logging_resource_adaptor_impl(
  std::shared_ptr<rapids_logger::logger> logger,
  device_async_resource_ref upstream,
  bool auto_flush)
  : logger_{std::move(logger)}, upstream_{upstream}
{
  if (auto_flush) { logger_->flush_on(rapids_logger::level_enum::info); }
  logger_->set_pattern("%v");
  logger_->info(header());
  logger_->set_pattern("%t,%H:%M:%S.%f,%v");
}

void* logging_resource_adaptor_impl::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  auto const stream = cuda_stream_view{};
  try {
    auto const ptr = upstream_.allocate(stream, bytes, alignment);
    logger_->info("allocate,%p,%zu,%s", ptr, bytes, rmm::detail::format_stream(stream));
    return ptr;
  } catch (...) {
    logger_->info("allocate failure,%p,%zu,%s", nullptr, bytes, rmm::detail::format_stream(stream));
    throw;
  }
}

void logging_resource_adaptor_impl::deallocate_sync(void* ptr,
                                                    std::size_t bytes,
                                                    std::size_t alignment) noexcept
{
  auto const stream = cuda_stream_view{};
  logger_->info("free,%p,%zu,%s", ptr, bytes, rmm::detail::format_stream(stream));
  upstream_.deallocate(stream, ptr, bytes, alignment);
}

void* logging_resource_adaptor_impl::allocate(cuda::stream_ref stream,
                                              std::size_t bytes,
                                              std::size_t alignment)
{
  try {
    auto const ptr = upstream_.allocate(stream, bytes, alignment);
    logger_->info("allocate,%p,%zu,%s", ptr, bytes, rmm::detail::format_stream(stream));
    return ptr;
  } catch (...) {
    logger_->info("allocate failure,%p,%zu,%s", nullptr, bytes, rmm::detail::format_stream(stream));
    throw;
  }
}

void logging_resource_adaptor_impl::deallocate(cuda::stream_ref stream,
                                               void* ptr,
                                               std::size_t bytes,
                                               std::size_t alignment) noexcept
{
  logger_->info("free,%p,%zu,%s", ptr, bytes, rmm::detail::format_stream(stream));
  upstream_.deallocate(stream, ptr, bytes, alignment);
}

rmm::device_async_resource_ref logging_resource_adaptor_impl::get_upstream_resource() const noexcept
{
  return rmm::device_async_resource_ref{
    const_cast<cuda::mr::any_resource<cuda::mr::device_accessible>&>(upstream_)};
}

void logging_resource_adaptor_impl::flush() { logger_->flush(); }

std::string logging_resource_adaptor_impl::header() const
{
  return std::string{"Thread,Time,Action,Pointer,Size,Stream"};
}

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
