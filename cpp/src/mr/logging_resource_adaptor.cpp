/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/detail/format.hpp>
#include <rmm/mr/logging_resource_adaptor.hpp>

#include <cstdlib>

namespace RMM_NAMESPACE {
namespace mr {

namespace {

auto make_logger(std::ostream& stream)
{
  return std::make_shared<rapids_logger::logger>("RMM", stream);
}

auto make_logger(std::string const& filename)
{
  return std::make_shared<rapids_logger::logger>("RMM", filename);
}

auto make_logger(std::initializer_list<rapids_logger::sink_ptr> sinks)
{
  return std::make_shared<rapids_logger::logger>("RMM", sinks);
}

}  // namespace

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

logging_resource_adaptor::logging_resource_adaptor(device_async_resource_ref upstream,
                                                   std::string const& filename,
                                                   bool auto_flush)
  : cuda::mr::shared_resource<detail::logging_resource_adaptor_impl>(
      cuda::mr::make_shared_resource<detail::logging_resource_adaptor_impl>(
        make_logger(filename), upstream, auto_flush))
{
}

logging_resource_adaptor::logging_resource_adaptor(device_async_resource_ref upstream,
                                                   std::ostream& stream,
                                                   bool auto_flush)
  : cuda::mr::shared_resource<detail::logging_resource_adaptor_impl>(
      cuda::mr::make_shared_resource<detail::logging_resource_adaptor_impl>(
        make_logger(stream), upstream, auto_flush))
{
}

logging_resource_adaptor::logging_resource_adaptor(
  device_async_resource_ref upstream,
  std::initializer_list<rapids_logger::sink_ptr> sinks,
  bool auto_flush)
  : cuda::mr::shared_resource<detail::logging_resource_adaptor_impl>(
      cuda::mr::make_shared_resource<detail::logging_resource_adaptor_impl>(
        make_logger(sinks), upstream, auto_flush))
{
}

rmm::device_async_resource_ref logging_resource_adaptor::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

void logging_resource_adaptor::flush() { get().flush(); }

std::string logging_resource_adaptor::header() const { return get().header(); }

std::string logging_resource_adaptor::get_default_filename()
{
  auto* filename = std::getenv("RMM_LOG_FILE");
  RMM_EXPECTS(filename != nullptr,
              "RMM logging requested without an explicit file name, but RMM_LOG_FILE is unset");
  return std::string{filename};
}

// Begin legacy device_memory_resource compatibility layer
void* logging_resource_adaptor::do_allocate(std::size_t bytes, cuda_stream_view stream)
{
  return cuda::mr::shared_resource<detail::logging_resource_adaptor_impl>::allocate(
    stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

void logging_resource_adaptor::do_deallocate(void* ptr,
                                             std::size_t bytes,
                                             cuda_stream_view stream) noexcept
{
  cuda::mr::shared_resource<detail::logging_resource_adaptor_impl>::deallocate(
    stream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

bool logging_resource_adaptor::do_is_equal(device_memory_resource const& other) const noexcept
{
  if (this == &other) { return true; }
  auto const* cast = dynamic_cast<logging_resource_adaptor const*>(&other);
  if (cast == nullptr) { return false; }
  return static_cast<cuda::mr::shared_resource<detail::logging_resource_adaptor_impl> const&>(
           *this) ==
         static_cast<cuda::mr::shared_resource<detail::logging_resource_adaptor_impl> const&>(
           *cast);
}
// End legacy device_memory_resource compatibility layer

}  // namespace mr
}  // namespace RMM_NAMESPACE
