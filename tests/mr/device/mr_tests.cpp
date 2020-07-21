/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <rmm/mr/device/owning_wrapper.hpp>
#include "mr/device/cuda_memory_resource.hpp"
#include "mr/device/device_memory_resource.hpp"
#include "mr_test.hpp"

namespace rmm {
namespace test {
namespace {

using MRFactoryFunc = std::function<std::shared_ptr<rmm::mr::device_memory_resource>()>;

/// Encapsulates a `device_memory_resource` factory function and associated name
struct mr_factory {
  mr_factory(std::string const& name, MRFactoryFunc f) : name{name}, f{f} {}

  std::string name;  ///< Name to associate with tests that use this factory
  MRFactoryFunc f;   ///< Factory function that returns shared_ptr to `device_memory_resource`
                     ///< instance to use in test
};

/// Test fixture class value-parameterized on different `mr_factory`s
struct mr_test : public ::testing::TestWithParam<mr_factory> {
  void SetUp() override
  {
    auto factory = GetParam().f;
    mr           = factory();
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
  }

  void TearDown() override { EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream)); };

  std::shared_ptr<rmm::mr::device_memory_resource> mr;  ///< Pointer to resource to use in tests
  cudaStream_t stream;
};

/// MR factory functions
auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

auto make_managed() { return std::make_shared<rmm::mr::managed_memory_resource>(); }

auto make_cnmem() { return std::make_shared<rmm::mr::cnmem_memory_resource>(); }

auto make_cnmem_managed() { return std::make_shared<rmm::mr::cnmem_managed_memory_resource>(); }

auto make_pool()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_cuda());
}

auto make_fixed_size()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::fixed_size_memory_resource>(make_cuda());
}

auto make_multisize()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::fixed_multisize_memory_resource>(make_cuda());
}

auto make_hybrid()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::hybrid_memory_resource>(
    std::make_tuple(make_multisize(), make_pool()));
}

INSTANTIATE_TEST_CASE_P(ResourceTests,
                        mr_test,
                        ::testing::Values(mr_factory{"CUDA", &make_cuda},
                                          mr_factory{"Managed", &make_managed},
                                          mr_factory{"CNMEM", &make_cnmem},
                                          mr_factory{"CNMEM_Managed", &make_cnmem_managed},
                                          mr_factory{"Pool", &make_pool},
                                          mr_factory{"Hybrid", &make_hybrid}),
                        [](auto const& info) { return info.param.name; });

TEST(DefaultTest, UseDefaultResource) { test_get_default_resource(); }

TEST_P(mr_test, SetDefaultResource)
{
  rmm::mr::device_memory_resource* old{nullptr};
  EXPECT_NO_THROW(old = rmm::mr::set_default_resource(this->mr.get()));
  EXPECT_NE(nullptr, old);

  test_get_default_resource();  // test allocating with the new default resource

  // setting default resource w/ nullptr should reset to initial
  EXPECT_NO_THROW(rmm::mr::set_default_resource(nullptr));
  EXPECT_TRUE(old->is_equal(*rmm::mr::get_default_resource()));
}

TEST_P(mr_test, SelfEquality) { EXPECT_TRUE(this->mr->is_equal(*this->mr)); }

TEST_P(mr_test, AllocateDefaultStream)
{
  test_various_allocations(this->mr.get(), cudaStreamDefault);
}

TEST_P(mr_test, AllocateOnStream) { test_various_allocations(this->mr.get(), this->stream); }

TEST_P(mr_test, RandomAllocations) { test_random_allocations(this->mr.get()); }

TEST_P(mr_test, RandomAllocationsStream)
{
  test_random_allocations(this->mr.get(), 100, 5_MiB, this->stream);
}

TEST_P(mr_test, MixedRandomAllocationFree)
{
  test_mixed_random_allocation_free(this->mr.get(), 5_MiB, cudaStreamDefault);
}

TEST_P(mr_test, MixedRandomAllocationFreeStream)
{
  test_mixed_random_allocation_free(this->mr.get(), 5_MiB, this->stream);
}

TEST_P(mr_test, GetMemInfo)
{
  if (this->mr->supports_get_mem_info()) {
    std::pair<std::size_t, std::size_t> mem_info;
    EXPECT_NO_THROW(mem_info = this->mr->get_mem_info(0));
    std::size_t allocation_size = 16 * 256;
    void* ptr;
    EXPECT_NO_THROW(ptr = this->mr->allocate(allocation_size));
    EXPECT_NO_THROW(mem_info = this->mr->get_mem_info(0));
    EXPECT_TRUE(mem_info.first >= allocation_size);
    EXPECT_NO_THROW(this->mr->deallocate(ptr, allocation_size));
  }
}
}  // namespace
}  // namespace test
}  // namespace rmm
