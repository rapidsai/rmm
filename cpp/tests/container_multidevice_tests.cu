/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include "device_check_resource_adaptor.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

template <typename ContainerType>
struct ContainerMultiDeviceTest : public ::testing::Test {};

using containers =
  ::testing::Types<rmm::device_buffer, rmm::device_uvector<int>, rmm::device_scalar<int>>;

TYPED_TEST_SUITE(ContainerMultiDeviceTest, containers);

TYPED_TEST(ContainerMultiDeviceTest, CreateDestroyDifferentActiveDevice)
{
  // Get the number of cuda devices
  int num_devices = rmm::get_num_cuda_devices();

  // only run on multidevice systems
  if (num_devices >= 2) {
    rmm::cuda_set_device_raii dev{rmm::cuda_device_id{0}};
    auto orig_mr  = rmm::mr::get_current_device_resource_ref();
    auto check_mr = device_check_resource_adaptor{orig_mr};
    rmm::mr::set_current_device_resource_ref(check_mr);

    {
      if constexpr (std::is_same_v<TypeParam, rmm::device_scalar<int>>) {
        auto buf = TypeParam(rmm::cuda_stream_view{});
        RMM_ASSERT_CUDA_SUCCESS(cudaSetDevice(1));  // force dtor with different active device
      } else {
        auto buf = TypeParam(128, rmm::cuda_stream_view{});
        RMM_ASSERT_CUDA_SUCCESS(cudaSetDevice(1));  // force dtor with different active device
      }
    }

    RMM_ASSERT_CUDA_SUCCESS(cudaSetDevice(0));
    rmm::mr::set_current_device_resource_ref(orig_mr);
  }
}

TYPED_TEST(ContainerMultiDeviceTest, CreateMoveDestroyDifferentActiveDevice)
{
  // Get the number of cuda devices
  int num_devices = rmm::get_num_cuda_devices();

  // only run on multidevice systems
  if (num_devices >= 2) {
    rmm::cuda_set_device_raii dev{rmm::cuda_device_id{0}};
    auto orig_mr  = rmm::mr::get_current_device_resource_ref();
    auto check_mr = device_check_resource_adaptor{orig_mr};
    rmm::mr::set_current_device_resource_ref(check_mr);

    {
      auto buf_1 = []() {
        if constexpr (std::is_same_v<TypeParam, rmm::device_scalar<int>>) {
          return TypeParam(rmm::cuda_stream_view{});
        } else {
          return TypeParam(128, rmm::cuda_stream_view{});
        }
      }();

      {
        if constexpr (std::is_same_v<TypeParam, rmm::device_scalar<int>>) {
          // device_vector does not have a constructor that takes a stream
          auto buf_0 = TypeParam(rmm::cuda_stream_view{});
          buf_1      = std::move(buf_0);
        } else {
          auto buf_0 = TypeParam(128, rmm::cuda_stream_view{});
          buf_1      = std::move(buf_0);
        }
      }

      RMM_ASSERT_CUDA_SUCCESS(cudaSetDevice(1));  // force dtor with different active device
    }

    RMM_ASSERT_CUDA_SUCCESS(cudaSetDevice(0));
    rmm::mr::set_current_device_resource_ref(orig_mr);
  }
}

TYPED_TEST(ContainerMultiDeviceTest, ResizeDifferentActiveDevice)
{
  // Get the number of cuda devices
  int num_devices = rmm::get_num_cuda_devices();

  // only run on multidevice systems
  if (num_devices >= 2) {
    rmm::cuda_set_device_raii dev{rmm::cuda_device_id{0}};
    auto orig_mr  = rmm::mr::get_current_device_resource_ref();
    auto check_mr = device_check_resource_adaptor{orig_mr};
    rmm::mr::set_current_device_resource_ref(check_mr);

    if constexpr (not std::is_same_v<TypeParam, rmm::device_scalar<int>>) {
      auto buf = TypeParam(128, rmm::cuda_stream_view{});
      RMM_ASSERT_CUDA_SUCCESS(cudaSetDevice(1));  // force resize with different active device
      buf.resize(1024, rmm::cuda_stream_view{});
    }

    RMM_ASSERT_CUDA_SUCCESS(cudaSetDevice(0));
    rmm::mr::set_current_device_resource_ref(orig_mr);
  }
}

TYPED_TEST(ContainerMultiDeviceTest, ShrinkDifferentActiveDevice)
{
  // Get the number of cuda devices
  int num_devices = rmm::get_num_cuda_devices();

  // only run on multidevice systems
  if (num_devices >= 2) {
    rmm::cuda_set_device_raii dev{rmm::cuda_device_id{0}};
    auto orig_mr  = rmm::mr::get_current_device_resource_ref();
    auto check_mr = device_check_resource_adaptor{orig_mr};
    rmm::mr::set_current_device_resource_ref(check_mr);

    if constexpr (not std::is_same_v<TypeParam, rmm::device_scalar<int>>) {
      auto buf = TypeParam(128, rmm::cuda_stream_view{});
      RMM_ASSERT_CUDA_SUCCESS(cudaSetDevice(1));  // force resize with different active device
      buf.resize(64, rmm::cuda_stream_view{});
      buf.shrink_to_fit(rmm::cuda_stream_view{});
    }

    RMM_ASSERT_CUDA_SUCCESS(cudaSetDevice(0));
    rmm::mr::set_current_device_resource_ref(orig_mr);
  }
}
