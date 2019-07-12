/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#include "gtest/gtest.h"
#include "test_fixtures.h"
#include "rmm/rmm.h"

#include <numeric>

// Helper macros to simplify testing for success or failure
#define ASSERT_SUCCESS(res) ASSERT_EQ(RMM_SUCCESS, (res));
#define ASSERT_FAILURE(res) ASSERT_NE(RMM_SUCCESS, (res));

cudaStream_t stream;

template <typename T>
struct MemoryManagerTest : public ::testing::Test {
  static rmmAllocationMode_t allocationMode() { return T::alloc_mode; }

  static void SetUpTestCase() {
    ASSERT_FALSE(rmmIsInitialized(0));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    rmmOptions_t options{};
    options.allocation_mode = allocationMode();
    ASSERT_SUCCESS(rmmInitialize(&options));
    rmmOptions_t options_set;
    // verify initialized
    ASSERT_TRUE(rmmIsInitialized(&options_set));
    // verify initialized options
    ASSERT_EQ(options_set.allocation_mode, options.allocation_mode);
    ASSERT_EQ(options_set.initial_pool_size, options.initial_pool_size);
    ASSERT_EQ(options_set.enable_logging, options.enable_logging);
  }

  static void TearDownTestCase() {
    ASSERT_SUCCESS(rmmFinalize());
    ASSERT_FALSE(rmmIsInitialized(0));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  }

  // some useful allocation sizes
  static const size_t size_word{4};
  static const size_t size_kb{size_t{1} << 10};
  static const size_t size_mb{size_t{1} << 20};
  static const size_t size_gb{size_t{1} << 30};
  static const size_t size_tb{size_t{1} << 40};
  static const size_t size_pb{size_t{1} << 50};
};


template <rmmAllocationMode_t mode>
struct ModeType {
    static constexpr rmmAllocationMode_t alloc_mode{mode};
};

using allocation_modes = ::testing::Types< ModeType<CudaDefaultAllocation>,
                                           ModeType<PoolAllocation>,
                                           ModeType<CudaManagedMemory>,
                                           ModeType<static_cast<rmmAllocationMode_t>(PoolAllocation | CudaManagedMemory)>
                                         >;



TYPED_TEST_CASE(MemoryManagerTest, allocation_modes);

// Init / Finalize tests

TYPED_TEST(MemoryManagerTest, Initialize) {
    
    // Initialized in Fixture class.
    rmmOptions_t options;
    ASSERT_TRUE(rmmIsInitialized(&options));
}

TYPED_TEST(MemoryManagerTest, Finalize) {
    // Empty because handled in Fixture class.
}

// zero size tests

TYPED_TEST(MemoryManagerTest, AllocateZeroBytes) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC(&a, 0, stream) );
}

TYPED_TEST(MemoryManagerTest, NullPtrAllocateZeroBytes) {
    char ** p{nullptr};
    ASSERT_SUCCESS( RMM_ALLOC(p, 0, stream) );
}

// Bad argument tests

TYPED_TEST(MemoryManagerTest, NullPtrInvalidArgument) {
    char ** p{nullptr};
    rmmError_t res = RMM_ALLOC(p, 4, stream);
    ASSERT_FAILURE(res);
    ASSERT_EQ(RMM_ERROR_INVALID_ARGUMENT, res);
}

// Simple allocation / free tests

TYPED_TEST(MemoryManagerTest, AllocateWord) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_word, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MemoryManagerTest, AllocateKB) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_kb, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MemoryManagerTest, AllocateMB) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_mb, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MemoryManagerTest, AllocateGB) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_gb, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MemoryManagerTest, AllocateTB) {
    char *a = 0;
    size_t freeBefore = 0, totalBefore = 0;
    ASSERT_SUCCESS( rmmGetInfo(&freeBefore, &totalBefore, stream) );

    if ((this->allocationMode() & CudaManagedMemory) || 
        (this->size_tb < freeBefore)) {
        // TODO investigate and fix this
        //ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_tb, stream) );
    }
    else {
        ASSERT_FAILURE( RMM_ALLOC(&a, this->size_tb, stream) );
    }
    
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MemoryManagerTest, AllocateTooMuch) {
    char *a = 0;
    ASSERT_FAILURE( RMM_ALLOC(&a, this->size_pb, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MemoryManagerTest, FreeZero) {
    ASSERT_SUCCESS( RMM_FREE(0, stream) );
}

// Reallocation tests

TYPED_TEST(MemoryManagerTest, ReallocateSmaller) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_mb, stream) );
    ASSERT_SUCCESS( RMM_REALLOC(&a, this->size_mb / 2, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MemoryManagerTest, ReallocateMuchSmaller) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_gb, stream) );
    ASSERT_SUCCESS( RMM_REALLOC(&a, this->size_kb, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MemoryManagerTest, ReallocateLarger) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_mb, stream) );
    ASSERT_SUCCESS( RMM_REALLOC(&a, this->size_mb * 2, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MemoryManagerTest, ReallocateMuchLarger) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_kb, stream) );
    ASSERT_SUCCESS( RMM_REALLOC(&a, this->size_gb, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MemoryManagerTest, GetInfo) {
    size_t freeBefore = 0, totalBefore = 0;
    ASSERT_SUCCESS( rmmGetInfo(&freeBefore, &totalBefore, stream) );
    ASSERT_GE(freeBefore, 0);
    ASSERT_GE(totalBefore, 0);
}

TYPED_TEST(MemoryManagerTest, AllocationOffset) {
    char *a = nullptr, *b = nullptr;
    ptrdiff_t offset = -1;
    ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_mb, stream) );
    ASSERT_SUCCESS( RMM_ALLOC(&b, this->size_mb, stream) );

    ASSERT_SUCCESS( rmmGetAllocationOffset(&offset, a, stream) );
    ASSERT_GE(offset, 0);

    ASSERT_SUCCESS( rmmGetAllocationOffset(&offset, b, stream) );
    ASSERT_GE(offset, 0);

    ASSERT_SUCCESS( RMM_FREE(a, stream) );
    ASSERT_SUCCESS( RMM_FREE(b, stream) );
}

template <typename T>
struct MultiGPUMemoryManagerTest : public MemoryManagerTest<T> {
    static void SetUpTestCase() {
        ASSERT_FALSE(rmmIsInitialized(0));
        ASSERT_EQ( cudaSuccess, cudaStreamCreate(&stream) );
        int device_count{};
        ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&device_count));
        std::vector<int> devices(device_count);
        std::iota(begin(devices), end(devices),0);
        rmmOptions_t options{};
        options.allocation_mode = T::alloc_mode;
        options.devices = devices.data();
        options.num_devices = devices.size();
        ASSERT_SUCCESS( rmmInitialize(&options) );
        rmmOptions_t options_set;
        // verify initialized
        ASSERT_TRUE(rmmIsInitialized(&options_set));
        // verify initialized options
        ASSERT_EQ(options_set.allocation_mode, options.allocation_mode);
        ASSERT_EQ(options_set.initial_pool_size, options.initial_pool_size);
        ASSERT_EQ(options_set.enable_logging, options.enable_logging);
    }
};

TYPED_TEST_CASE(MultiGPUMemoryManagerTest, allocation_modes);

// Init / Finalize tests

TYPED_TEST(MultiGPUMemoryManagerTest, Initialize) {
    
    // Initialized in Fixture class.
    rmmOptions_t options;
    ASSERT_TRUE(rmmIsInitialized(&options));
}

TYPED_TEST(MultiGPUMemoryManagerTest, Finalize) {
    // Empty because handled in Fixture class.
}

// zero size tests

TYPED_TEST(MultiGPUMemoryManagerTest, AllocateZeroBytes) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC(&a, 0, stream) );
}

TYPED_TEST(MultiGPUMemoryManagerTest, NullPtrAllocateZeroBytes) {
    char ** p{nullptr};
    ASSERT_SUCCESS( RMM_ALLOC(p, 0, stream) );
}

// Bad argument tests

TYPED_TEST(MultiGPUMemoryManagerTest, NullPtrInvalidArgument) {
    char ** p{nullptr};
    rmmError_t res = RMM_ALLOC(p, 4, stream);
    ASSERT_FAILURE(res);
    ASSERT_EQ(RMM_ERROR_INVALID_ARGUMENT, res);
}

// Simple allocation / free tests

TYPED_TEST(MultiGPUMemoryManagerTest, AllocateWord) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_word, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MultiGPUMemoryManagerTest, AllocateKB) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_kb, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MultiGPUMemoryManagerTest, AllocateMB) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_mb, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MultiGPUMemoryManagerTest, AllocateGB) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_gb, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MultiGPUMemoryManagerTest, AllocateTB) {
    char *a = 0;
    size_t freeBefore = 0, totalBefore = 0;
    ASSERT_SUCCESS( rmmGetInfo(&freeBefore, &totalBefore, stream) );

    if ((this->allocationMode() & CudaManagedMemory) || 
        (this->size_tb < freeBefore)) {
        // TODO investigate and fix this
        //ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_tb, stream) );
    }
    else {
        ASSERT_FAILURE( RMM_ALLOC(&a, this->size_tb, stream) );
    }
    
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MultiGPUMemoryManagerTest, AllocateTooMuch) {
    char *a = 0;
    ASSERT_FAILURE( RMM_ALLOC(&a, this->size_pb, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MultiGPUMemoryManagerTest, FreeZero) {
    ASSERT_SUCCESS( RMM_FREE(0, stream) );
}

// Reallocation tests

TYPED_TEST(MultiGPUMemoryManagerTest, ReallocateSmaller) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_mb, stream) );
    ASSERT_SUCCESS( RMM_REALLOC(&a, this->size_mb / 2, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MultiGPUMemoryManagerTest, ReallocateMuchSmaller) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_gb, stream) );
    ASSERT_SUCCESS( RMM_REALLOC(&a, this->size_kb, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MultiGPUMemoryManagerTest, ReallocateLarger) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_mb, stream) );
    ASSERT_SUCCESS( RMM_REALLOC(&a, this->size_mb * 2, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MultiGPUMemoryManagerTest, ReallocateMuchLarger) {
    char *a = 0;
    ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_kb, stream) );
    ASSERT_SUCCESS( RMM_REALLOC(&a, this->size_gb, stream) );
    ASSERT_SUCCESS( RMM_FREE(a, stream) );
}

TYPED_TEST(MultiGPUMemoryManagerTest, GetInfo) {
    size_t freeBefore = 0, totalBefore = 0;
    ASSERT_SUCCESS( rmmGetInfo(&freeBefore, &totalBefore, stream) );
    ASSERT_GE(freeBefore, 0);
    ASSERT_GE(totalBefore, 0);
}

TYPED_TEST(MultiGPUMemoryManagerTest, AllocationOffset) {
    char *a = nullptr, *b = nullptr;
    ptrdiff_t offset = -1;
    ASSERT_SUCCESS( RMM_ALLOC(&a, this->size_mb, stream) );
    ASSERT_SUCCESS( RMM_ALLOC(&b, this->size_mb, stream) );

    ASSERT_SUCCESS( rmmGetAllocationOffset(&offset, a, stream) );
    ASSERT_GE(offset, 0);

    ASSERT_SUCCESS( rmmGetAllocationOffset(&offset, b, stream) );
    ASSERT_GE(offset, 0);

    ASSERT_SUCCESS( RMM_FREE(a, stream) );
    ASSERT_SUCCESS( RMM_FREE(b, stream) );
}