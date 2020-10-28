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

#include <memory>

#include <gtest/gtest.h>
#include <rmm/mr/device/polymorphic_allocator.hpp>
#include "mr_test.hpp"

namespace rmm {
namespace test {
namespace {

struct allocator_test : public mr_test {
};

template <typename Allocator>
void test_allocator(Allocator const& alloc){

}

template <typename Allocator>
void test_stream_ordered_allocator(Allocator const& alloc){
    auto allocator = Allocator{alloc};
    int * p = allocator.allocate(1000, cudaStream_t{0});
    EXPECT_NE(p, nullptr);
    EXPECT_NO_THROW(allocator.deallocate(p, 1000, cudaStream_t{0}));
}

TEST(first, first) { 
    rmm::mr::polymorphic_allocator<int> allocator{}; 
    test_stream_ordered_allocator(allocator);
}

}  // namespace
}  // namespace test
}  // namespace rmm
