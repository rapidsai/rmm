/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <rmm/error.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

// explicit instantiation for test coverage purposes
template class rmm::mr::binning_memory_resource<rmm::mr::cuda_memory_resource>;

namespace rmm::test {

using cuda_mr    = rmm::mr::cuda_memory_resource;
using binning_mr = rmm::mr::binning_memory_resource<cuda_mr>;

TEST(BinningTest, ThrowOnNullUpstream)
{
  auto construct_nullptr = []() { binning_mr mr{nullptr}; };
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);
}

TEST(BinningTest, ExplicitBinMR)
{
  cuda_mr cuda{};
  binning_mr mr{&cuda};
  mr.add_bin(1024, &cuda);
  auto* ptr = mr.allocate(512);
  EXPECT_NE(ptr, nullptr);
  mr.deallocate(ptr, 512);
}

}  // namespace rmm::test
