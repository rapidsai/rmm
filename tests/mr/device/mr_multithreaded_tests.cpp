/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "mr_test.hpp"

#include <gtest/gtest.h>

#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thread>
#include <vector>

namespace rmm {
namespace test {
namespace {

struct mr_test_mt : public mr_test {
};

INSTANTIATE_TEST_CASE_P(MultiThreadResourceTests,
                        mr_test_mt,
                        ::testing::Values(mr_factory{"CUDA", &make_cuda},
                                          mr_factory{"Managed", &make_managed},
                                          mr_factory{"Pool", &make_pool},
                                          mr_factory{"Arena", &make_arena},
                                          mr_factory{"Binning", &make_binning}),
                        [](auto const& info) { return info.param.name; });

template <typename Task, typename... Arguments>
void spawn_n(std::size_t num_threads, Task task, Arguments&&... args)
{
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i)
    threads.emplace_back(std::thread(task, std::forward<Arguments>(args)...));

  for (auto& t : threads)
    t.join();
}

template <typename Task, typename... Arguments>
void spawn(Task task, Arguments&&... args)
{
  spawn_n(4, task, std::forward<Arguments>(args)...);
}

TEST(DefaultTest, UseCurrentDeviceResource_mt) { spawn(test_get_current_device_resource); }

TEST(DefaultTest, CurrentDeviceResourceIsCUDA_mt)
{
  spawn([]() {
    EXPECT_NE(nullptr, rmm::mr::get_current_device_resource());
    EXPECT_TRUE(rmm::mr::get_current_device_resource()->is_equal(rmm::mr::cuda_memory_resource{}));
  });
}

TEST(DefaultTest, GetCurrentDeviceResource_mt)
{
  spawn([]() {
    rmm::mr::device_memory_resource* mr;
    EXPECT_NO_THROW(mr = rmm::mr::get_current_device_resource());
    EXPECT_NE(nullptr, mr);
    EXPECT_TRUE(mr->is_equal(rmm::mr::cuda_memory_resource{}));
  });
}

TEST_P(mr_test_mt, SetCurrentDeviceResource_mt)
{
  // single thread changes default resource, then multiple threads use it

  rmm::mr::device_memory_resource* old{nullptr};
  EXPECT_NO_THROW(old = rmm::mr::set_current_device_resource(this->mr.get()));
  EXPECT_NE(nullptr, old);

  spawn([mr = this->mr.get()]() {
    EXPECT_EQ(mr, rmm::mr::get_current_device_resource());
    test_get_current_device_resource();  // test allocating with the new default resource
  });

  // setting default resource w/ nullptr should reset to initial
  EXPECT_NO_THROW(rmm::mr::set_current_device_resource(nullptr));
  EXPECT_TRUE(old->is_equal(*rmm::mr::get_current_device_resource()));
}

TEST_P(mr_test_mt, SetCurrentDeviceResourcePerThread_mt)
{
  int num_devices;
  RMM_CUDA_TRY(cudaGetDeviceCount(&num_devices));

  std::vector<std::thread> threads;
  threads.reserve(num_devices);
  for (int i = 0; i < num_devices; ++i) {
    threads.emplace_back(std::thread{
      [mr = this->mr.get()](auto dev_id) {
        RMM_CUDA_TRY(cudaSetDevice(dev_id));
        rmm::mr::device_memory_resource* old;
        EXPECT_NO_THROW(old = rmm::mr::set_current_device_resource(mr));
        EXPECT_NE(nullptr, old);
        // initial resource for this device should be CUDA mr
        EXPECT_TRUE(old->is_equal(rmm::mr::cuda_memory_resource{}));
        // get_current_device_resource should equal the resource we just set
        EXPECT_EQ(mr, rmm::mr::get_current_device_resource());
        // Setting current dev resource to nullptr should reset to cuda MR and return the MR we
        // previously set
        EXPECT_NO_THROW(old = rmm::mr::set_current_device_resource(nullptr));
        EXPECT_NE(nullptr, old);
        EXPECT_EQ(old, mr);
        EXPECT_TRUE(
          rmm::mr::get_current_device_resource()->is_equal(rmm::mr::cuda_memory_resource{}));
      },
      i});
  }

  for (auto& t : threads)
    t.join();
}

TEST_P(mr_test_mt, AllocateDefaultStream)
{
  spawn(test_various_allocations, this->mr.get(), cudaStream_t{cudaStreamDefault});
}

TEST_P(mr_test_mt, AllocateOnStream)
{
  spawn(test_various_allocations, this->mr.get(), this->stream);
}

TEST_P(mr_test_mt, RandomAllocationsDefaultStream)
{
  spawn(test_random_allocations, this->mr.get(), 100, 5_MiB, cudaStream_t{cudaStreamDefault});
}

TEST_P(mr_test_mt, RandomAllocationsStream)
{
  spawn(test_random_allocations, this->mr.get(), 100, 5_MiB, this->stream);
}

TEST_P(mr_test_mt, MixedRandomAllocationFreeDefaultStream)
{
  spawn(test_mixed_random_allocation_free, this->mr.get(), 5_MiB, cudaStream_t{cudaStreamDefault});
}

TEST_P(mr_test_mt, MixedRandomAllocationFreeStream)
{
  spawn(test_mixed_random_allocation_free, this->mr.get(), 5_MiB, this->stream);
}

void allocate_loop(rmm::mr::device_memory_resource* mr,
                   std::size_t num_allocations,
                   std::list<allocation>& allocations,
                   std::mutex& mtx,
                   cudaStream_t stream)
{
  constexpr std::size_t max_size{1_MiB};

  std::default_random_engine generator;
  std::uniform_int_distribution<std::size_t> size_distribution(1, max_size);

  for (std::size_t i = 0; i < num_allocations; ++i) {
    size_t size = size_distribution(generator);
    void* ptr{};
    EXPECT_NO_THROW(ptr = mr->allocate(size, stream));
    {
      std::lock_guard<std::mutex> lock(mtx);
      allocations.emplace_back(ptr, size);
    }
  }
}

void deallocate_loop(rmm::mr::device_memory_resource* mr,
                     std::size_t num_allocations,
                     std::list<allocation>& allocations,
                     std::mutex& mtx,
                     cudaStream_t stream)
{
  for (std::size_t i = 0; i < num_allocations;) {
    std::lock_guard<std::mutex> lock(mtx);
    if (allocations.empty())
      continue;
    else {
      i++;
      allocation alloc = allocations.front();
      allocations.pop_front();
      EXPECT_NO_THROW(mr->deallocate(alloc.p, alloc.size, stream));
    }
  }
}

void test_allocate_free_different_threads(rmm::mr::device_memory_resource* mr,
                                          cudaStream_t streamA,
                                          cudaStream_t streamB)
{
  constexpr std::size_t num_allocations{100};

  std::mutex mtx;
  std::list<allocation> allocations;

  std::thread producer(
    allocate_loop, mr, num_allocations, std::ref(allocations), std::ref(mtx), streamA);

  std::thread consumer(
    deallocate_loop, mr, num_allocations, std::ref(allocations), std::ref(mtx), streamB);

  producer.join();
  consumer.join();
}

TEST_P(mr_test_mt, AllocFreeDifferentThreadsDefaultStream)
{
  test_allocate_free_different_threads(
    this->mr.get(), cudaStream_t{cudaStreamDefault}, cudaStream_t{cudaStreamDefault});
}

TEST_P(mr_test_mt, AllocFreeDifferentThreadsSameStream)
{
  test_allocate_free_different_threads(this->mr.get(), this->stream, this->stream);
}

TEST_P(mr_test_mt, AllocFreeDifferentThreadsDifferentStream)
{
  cudaStream_t streamB{};
  EXPECT_EQ(cudaSuccess, cudaStreamCreate(&streamB));
  test_allocate_free_different_threads(this->mr.get(), this->stream, streamB);
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(streamB));
  EXPECT_EQ(cudaSuccess, cudaStreamDestroy(streamB));
}

}  // namespace
}  // namespace test
}  // namespace rmm
