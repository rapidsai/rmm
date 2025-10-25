/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
    if (is_correct_device) { return get_upstream_resource().allocate(stream, bytes); }
    return nullptr;
  }

  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) noexcept override
  {
    bool const is_correct_device = check_device_id();
    EXPECT_TRUE(is_correct_device);
    if (is_correct_device) { get_upstream_resource().deallocate(stream, ptr, bytes); }
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
