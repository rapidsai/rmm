/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
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

logging_resource_adaptor::logging_resource_adaptor(device_async_resource_ref upstream,
                                                   std::string const& filename,
                                                   bool auto_flush)
  : shared_base(cuda::mr::make_shared_resource<detail::logging_resource_adaptor_impl>(
      make_logger(filename), upstream, auto_flush))
{
}

logging_resource_adaptor::logging_resource_adaptor(device_async_resource_ref upstream,
                                                   std::ostream& stream,
                                                   bool auto_flush)
  : shared_base(cuda::mr::make_shared_resource<detail::logging_resource_adaptor_impl>(
      make_logger(stream), upstream, auto_flush))
{
}

logging_resource_adaptor::logging_resource_adaptor(
  device_async_resource_ref upstream,
  std::initializer_list<rapids_logger::sink_ptr> sinks,
  bool auto_flush)
  : shared_base(cuda::mr::make_shared_resource<detail::logging_resource_adaptor_impl>(
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
  return shared_base::allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

void logging_resource_adaptor::do_deallocate(void* ptr,
                                             std::size_t bytes,
                                             cuda_stream_view stream) noexcept
{
  shared_base::deallocate(stream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

bool logging_resource_adaptor::do_is_equal(device_memory_resource const& other) const noexcept
{
  if (this == &other) { return true; }
  auto const* cast = dynamic_cast<logging_resource_adaptor const*>(&other);
  if (cast == nullptr) { return false; }
  return static_cast<shared_base const&>(*this) == static_cast<shared_base const&>(*cast);
}
// End legacy device_memory_resource compatibility layer

}  // namespace mr
}  // namespace RMM_NAMESPACE
