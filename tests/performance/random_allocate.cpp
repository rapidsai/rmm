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
#include <cuda_runtime_api.h>
#include <rmm/rmm.h>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <chrono>

using namespace std;

#define cudaSucceeded(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define rmmSucceeded(ans) { rmmAssert((ans), __FILE__, __LINE__); }
inline void rmmAssert(rmmError_t code, const char *file, int line, bool abort=true) {
    if (code != RMM_SUCCESS) {
        fprintf(stderr, "RMMassert: %s %d\n", file, line);
        if (abort) exit(code);
    }
}

cudaError_t (*gpuAlloc)(void** ptr, size_t sz) = cudaMalloc;
cudaError_t (*gpuFree)(void* ptr) = cudaFree;

cudaError_t _rmmAlloc(void **ptr, size_t sz) {
    rmmError_t res = RMM_ALLOC(ptr, sz, 0);
    rmmSucceeded(res);
    if (res != RMM_SUCCESS) return cudaErrorMemoryAllocation;
    return cudaSuccess;
}

cudaError_t _rmmFree(void *ptr) {
    rmmError_t res = RMM_FREE(ptr, 0);
    rmmSucceeded(res);
    if (res != RMM_SUCCESS) return cudaErrorMemoryAllocation;
    return cudaSuccess;
}

enum Allocator {
    cudaDefault = 0,
    rmmDefault,
    rmmManaged,
    rmmDefaultPool,
    rmmManagedPool
};

void setAllocator(const std::string alloc) {
    if (alloc == "cudaDefault") {
        gpuAlloc = cudaMalloc;
        gpuFree = cudaFree;
        return;
    }
    else {
        rmmOptions_t options{};
        if (alloc == "rmmManaged")
            options.allocation_mode = CudaManagedMemory;
        else if (alloc == "rmmDefaultPool")
            options.allocation_mode = PoolAllocation;
        else if (alloc == "rmmManagedPool")
            options.allocation_mode = 
                static_cast<rmmAllocationMode_t>(PoolAllocation | 
                                                 CudaManagedMemory);
        else assert(alloc == "rmmDefault");
        rmmInitialize(&options);
        gpuAlloc = _rmmAlloc;
        gpuFree = _rmmFree;
        return;
    }
}

using timer = std::chrono::high_resolution_clock;

// double precision time durations
template <typename T>
std::chrono::duration<double, std::micro>  microseconds(T&& t)
{
    return std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(t);
}

template <typename T>
std::chrono::duration<double, std::milli>  milliseconds(T&& t)
{
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t);
}

#define ALLOC_PROBABILITY 53
#define ALLOC 1
#define FREE 2
#define BAR_UNIT 80

#define MAX_BUFFER_SIZE_BYTE (1UL << 27)
#define MIN_BUFFER_SIZE_BYTE (1UL << 10)

#define KB (1UL << 10)
#define MB (1UL << 20)

// Using 88.7% of the memory to avoid OOM due to fragmentation
#define MEM_USAGE_PERCENTAGE 8870

#define SEED 123898464

int main(int argc, char** argv) {

    if (argc < 5) {
        printf("Usage: %s <allocator> <num allocations> <num unique sizes> <report average time every n allocations>\n", argv[0]);
        printf("Allocator is one of: cudaDefault, rmmDefault, rmmManaged, rmmDefaultPool, or rmmManagedPool\n");
        return 1;
    }

    setAllocator(argv[1]);

    int numAllocations = atoi(argv[2]);
    int numSizes = atoi(argv[3]);
    int averagePerN = atoi(argv[4]);
    printf("allocator: %s, numAllocations: %d, numSize: %d, report average every %d allocations\n", argv[1], numAllocations, numSizes, averagePerN);
    
    cudaStream_t st1;
    cudaSucceeded(cudaStreamCreate(&st1)); // Not used in this version

    //------------------------ creating some random sizes -------------------------//
    unsigned *sizes = (unsigned*) malloc(numSizes * sizeof(unsigned));
    srand(SEED);

    printf("Randomizing sizes between %luKB and %luKB bytes\n", MIN_BUFFER_SIZE_BYTE / KB, MAX_BUFFER_SIZE_BYTE / KB);
    for (int i = 0; i < numSizes; i ++) {
        sizes[i] = (rand() % (MAX_BUFFER_SIZE_BYTE - MIN_BUFFER_SIZE_BYTE)) + MIN_BUFFER_SIZE_BYTE;
    }
    //-----------------------------------------------------------------------------//


    //----------------------- create a bunch of allocation sizes -----------------//
    srand(SEED);
    unsigned* allocations = (unsigned*) malloc(numAllocations * sizeof(unsigned));
    void** buffers = (void**) malloc(numAllocations * sizeof(void*));
    long long unsigned totalAllocatedSize = 0;
    for (int i = 0; i < numAllocations; i ++) {
        allocations[i] = sizes[rand() % numSizes];
        buffers[i] = NULL;
        totalAllocatedSize += allocations[i];
    }
    //----------------------------------------------------------------------------//

    size_t totalMem, freeMem;
    cudaSucceeded(cudaMemGetInfo(&freeMem, &totalMem));
    
    //----------------- create the exact allocation-free plan --------------------//
    const int numAllocFree = numAllocations * 2;
    int* allocFrees = (int*)malloc(numAllocFree * sizeof(int));

    // This is the array the holds the valid allocations we currently have
    int* existingAllocations = (int*) malloc(numAllocations * sizeof(int));
    memset(existingAllocations, 0, numAllocations * sizeof(int));

    int allocCounter = 0; // Ignore the first allocation index (so that we can negate the index)
    size_t currentSize = 0;
    size_t maxSize = (size_t)(((freeMem / MB) * MEM_USAGE_PERCENTAGE) / 10000) * MB;
    int existingCounter = 0;

    // Printing the bar for max size, so user knows how to measure the usage based on the bar length
    printf("[                       max size: (%8luMB)               ]  [", freeMem / MB);
    for (size_t j = 0; j < ((maxSize / KB) * BAR_UNIT) / (maxSize / KB); j ++) {
        printf("-");
    }
    printf("]100.0%%\n");

    srand(SEED);
    for (int i = 0; i < numAllocFree; i++) {
        // Decide whether we want to allocate or free
        int allocOrFree = 0;
        int chance = rand() % 100;
        if (chance < ALLOC_PROBABILITY) {
            allocOrFree = ALLOC;
            if ((currentSize + allocations[allocCounter]) >= maxSize || allocCounter >= numAllocations) {
                allocOrFree = FREE;
            }
        }
        else {
            allocOrFree = FREE;
            if (currentSize <= 0) {
                allocOrFree = ALLOC;
            }
        }


        if (allocOrFree == ALLOC) {
            allocFrees[i] = allocCounter ++;

            // Record this allocation and move on
            existingAllocations[existingCounter++] = allocFrees[i];
            currentSize += allocations[allocFrees[i]];
            assert(currentSize < maxSize);
            printf("[%3d] Alloc index %4d with size %7luKB (current sum: %7luMB)[", i, allocFrees[i], allocations[allocFrees[i]] / KB, currentSize / MB);
            for (size_t j = 0; j < ((currentSize / KB) * BAR_UNIT) / (maxSize / KB); j ++) {
                printf("-");
            }
            double usage = (double)((currentSize / KB) * 100) / (double)(maxSize / KB);
            printf("]%3.1f%%\n", usage);
        }
        else {
            // Let's randomly pick one of the allocations that is not already free'd
            int allocationToFreeIndex = rand() % existingCounter;
            allocFrees[i] = existingAllocations[allocationToFreeIndex] * (-1);

            // Shift existingAllocations to remove the allocation
            for (int j = allocationToFreeIndex + 1; j < existingCounter; j ++) {
                existingAllocations[j - 1] = existingAllocations[j];
            }
            existingCounter --;
            currentSize -= allocations[allocFrees[i] * (-1)];
            printf("[%3d] Free  index %4d with size %7luKB (current sum: %7luMB)[", i, allocFrees[i], allocations[allocFrees[i] * (-1)] / KB, currentSize / MB);
            for (size_t j = 0; j < (currentSize * BAR_UNIT) / maxSize; j ++) {
                printf("-");
            }
            double usage = (double)((currentSize / KB) * 100) / (double)(maxSize / KB);
            printf("]%3.1f%%\n", usage);
        }

    }

    printf("Allocation-free plan is created. Executing the plan.\n");

    //int this_time_malloc, this_time_free, sum_time_malloc=0, sum_time_free=0, period_time_malloc=0, period_time_free=0;
    auto sum_time_malloc = std::chrono::duration<double>::zero();
    auto sum_time_free = std::chrono::duration<double>::zero();
    auto period_time_malloc = std::chrono::duration<double>::zero();
    auto period_time_free = std::chrono::duration<double>::zero();

    int period_count_malloc = 0;
    int period_count_free = 0;

    auto start = timer::now(); 
    // Do the first allocation outside the for, since its index is 0
    cudaSucceeded(gpuAlloc(&buffers[allocFrees[0]], allocations[allocFrees[0]]));
    auto end = timer::now();

    for (int i = 1; i < numAllocFree; i++) {
        if (allocFrees[i] > 0) {
            start = timer::now();
            if (gpuAlloc(&buffers[allocFrees[i]], allocations[allocFrees[i]]) != cudaSuccess) {
                printf("failed to allocate %dth allocation with size %luKB\n", i, allocations[allocFrees[i]] / KB);
                exit(1);
            }
            end = timer::now();
            std::chrono::duration<double> diff = end-start;
            sum_time_malloc += diff;

            period_count_malloc ++;
            period_time_malloc += diff;
            if (period_count_malloc >= averagePerN) {
                printf("Average malloc: %0.2f us\n",
                       (double)microseconds(period_time_malloc).count() / 
                            period_count_malloc);
                period_count_malloc = 0;
                period_time_malloc = std::chrono::duration<double>::zero();
            }
        }
        else {
            start = timer::now();
            cudaSucceeded(gpuFree(buffers[allocFrees[i] * (-1)]));
            end = timer::now();
            std::chrono::duration<double> diff = end-start;
            sum_time_free += diff;

            period_count_free ++;
            period_time_free += diff;
            if (period_count_free >= averagePerN) {
                printf("Average free: %0.2f us\n",
                       (double)microseconds(period_time_free).count() / 
                            period_count_free);
                period_count_free = 0;
                period_time_free = std::chrono::duration<double>::zero();
            }
        }
    }

    cudaSucceeded(cudaStreamSynchronize(st1));
    printf("sum allocation size: %llu MB\n", totalAllocatedSize / MB);
    printf("Average allocation size: %llu KB\n", 
           (totalAllocatedSize / numAllocations) / KB);
    printf("sum malloc: %f ms (average: %0.2f us)\n",
           (double)milliseconds(sum_time_malloc).count(),
           (double)microseconds(sum_time_malloc).count() / numAllocations);
    printf("sum free: %f ms (average: %0.2f us)\n",
           (double)milliseconds(sum_time_free).count(),
           (double)microseconds(sum_time_free).count() / numAllocations);

    return 0;
}
