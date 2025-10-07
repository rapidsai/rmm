/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <rmm/mr/device/device_memory_resource.hpp>

#include <gmock/gmock.h>

namespace rmm::test {

class mock_resource : public rmm::mr::device_memory_resource {
 public:
  MOCK_METHOD(void*, do_allocate, (std::size_t, cuda_stream_view), (override));
  MOCK_METHOD(void, do_deallocate, (void*, std::size_t, cuda_stream_view), (override));
  bool operator==(mock_resource const&) const noexcept { return true; }
  bool operator!=(mock_resource const&) const { return false; }
  friend void get_property(mock_resource const&, cuda::mr::device_accessible) noexcept {}
  using size_pair = std::pair<std::size_t, std::size_t>;
};

// static property checks
static_assert(
  rmm::detail::polyfill::async_resource_with<mock_resource, cuda::mr::device_accessible>);

}  // namespace rmm::test
