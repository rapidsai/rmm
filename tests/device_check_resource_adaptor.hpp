/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <gtest/gtest.h>

class device_check_resource_adaptor final : public rmm::mr::device_memory_resource {
 public:
  device_check_resource_adaptor(rmm::device_async_resource_ref upstream)
    : device_id{rmm::get_current_cuda_device()}, upstream_(upstream)
  {
  }

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept
  {
    return upstream_;
  }

 private:
  [[nodiscard]] bool check_device_id() const { return device_id == rmm::get_current_cuda_device(); }

  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override
  {
    bool const is_correct_device = check_device_id();
    EXPECT_TRUE(is_correct_device);
    if (is_correct_device) { return get_upstream_resource().allocate_async(bytes, stream); }
    return nullptr;
  }

  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) override
  {
    bool const is_correct_device = check_device_id();
    EXPECT_TRUE(is_correct_device);
    if (is_correct_device) { get_upstream_resource().deallocate_async(ptr, bytes, stream); }
  }

  [[nodiscard]] bool do_is_equal(
    rmm::mr::device_memory_resource const& other) const noexcept override
  {
    if (this == &other) { return true; }
    auto const* cast = dynamic_cast<device_check_resource_adaptor const*>(&other);
    if (cast == nullptr) { return false; }
    return get_upstream_resource() == cast->get_upstream_resource();
  }

  rmm::cuda_device_id device_id;
  rmm::device_async_resource_ref upstream_;
};
