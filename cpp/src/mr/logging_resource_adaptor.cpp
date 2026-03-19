/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/logging_resource_adaptor.hpp>

#include <cuda/memory_resource>

#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace RMM_NAMESPACE {
namespace mr {

logging_resource_adaptor::logging_resource_adaptor(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
  std::ostream& stream,
  bool auto_flush)
  : shared_base(cuda::mr::make_shared_resource<detail::logging_resource_adaptor_impl>(
      std::make_shared<rapids_logger::logger>("RMM", stream), std::move(upstream), auto_flush))
{
}

logging_resource_adaptor::logging_resource_adaptor(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
  std::initializer_list<rapids_logger::sink_ptr> sinks,
  bool auto_flush)
  : shared_base(cuda::mr::make_shared_resource<detail::logging_resource_adaptor_impl>(
      std::make_shared<rapids_logger::logger>("RMM", std::vector<rapids_logger::sink_ptr>(sinks)),
      std::move(upstream),
      auto_flush))
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

}  // namespace mr
}  // namespace RMM_NAMESPACE
