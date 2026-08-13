/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/detail/indexed_coalescing_free_list.hpp>
#include <rmm/mr/indexed_pool_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <benchmark/benchmark.h>
#include <benchmarks/utilities/simulated_memory_resource.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

struct allocation {
  void* pointer;
  std::size_t size;
};

std::size_t fragment_size(std::size_t index)
{
  constexpr std::size_t ALIGNMENT{rmm::CUDA_ALLOCATION_ALIGNMENT};
  return ((index % 8) + 1) * ALIGNMENT;
}

std::size_t fragmented_pool_size(std::size_t free_block_count)
{
  constexpr std::size_t SEPARATOR_SIZE{rmm::CUDA_ALLOCATION_ALIGNMENT};
  std::size_t result{};
  for (std::size_t i = 0; i < free_block_count; ++i) {
    result += fragment_size(i) + SEPARATOR_SIZE;
  }
  return result;
}

template <typename PoolResource>
void fragmented_best_fit_stream_ordered(benchmark::State& state, rmm::cuda_stream_view stream)
{
  auto const free_block_count = static_cast<std::size_t>(state.range(0));
  constexpr std::size_t SEPARATOR_SIZE{rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t REQUEST_SIZE{4 * rmm::CUDA_ALLOCATION_ALIGNMENT};
  auto const pool_size = fragmented_pool_size(free_block_count);

  rmm::mr::simulated_memory_resource upstream{pool_size};
  PoolResource pool{upstream, pool_size, pool_size};
  rmm::device_async_resource_ref resource{pool};
  std::vector<allocation> fragments;
  fragments.reserve(free_block_count);

  for (std::size_t i = 0; i < free_block_count; ++i) {
    auto const size = fragment_size(i);
    fragments.push_back({resource.allocate(stream, size, rmm::CUDA_ALLOCATION_ALIGNMENT), size});
    (void)resource.allocate(stream, SEPARATOR_SIZE, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  for (auto const& fragment : fragments) {
    resource.deallocate(stream, fragment.pointer, fragment.size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }

  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    auto* ptr = resource.allocate(stream, REQUEST_SIZE, rmm::CUDA_ALLOCATION_ALIGNMENT);
    benchmark::DoNotOptimize(ptr);
    resource.deallocate(stream, ptr, REQUEST_SIZE, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  state.SetItemsProcessed(state.iterations());
}

void BM_FragmentedBestFitLegacyStream(benchmark::State& state)
{
  fragmented_best_fit_stream_ordered<rmm::mr::pool_memory_resource>(state, rmm::cuda_stream_legacy);
}

void BM_FragmentedBestFitPerThreadStream(benchmark::State& state)
{
  fragmented_best_fit_stream_ordered<rmm::mr::pool_memory_resource>(state,
                                                                    rmm::cuda_stream_per_thread);
}

void BM_FragmentedBestFitIndexedLegacyStream(benchmark::State& state)
{
  fragmented_best_fit_stream_ordered<rmm::mr::indexed_pool_memory_resource>(
    state, rmm::cuda_stream_legacy);
}

void BM_FragmentedBestFitIndexedPerThreadStream(benchmark::State& state)
{
  fragmented_best_fit_stream_ordered<rmm::mr::indexed_pool_memory_resource>(
    state, rmm::cuda_stream_per_thread);
}

template <typename PoolResource>
void fragmented_best_fit_allocate_sync(benchmark::State& state)
{
  auto const free_block_count = static_cast<std::size_t>(state.range(0));
  constexpr std::size_t SEPARATOR_SIZE{rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t REQUEST_SIZE{4 * rmm::CUDA_ALLOCATION_ALIGNMENT};
  auto const pool_size = fragmented_pool_size(free_block_count);

  rmm::mr::simulated_memory_resource upstream{pool_size};
  PoolResource pool{upstream, pool_size, pool_size};
  std::vector<allocation> fragments;
  fragments.reserve(free_block_count);

  for (std::size_t i = 0; i < free_block_count; ++i) {
    auto const size = fragment_size(i);
    fragments.push_back({pool.allocate_sync(size), size});
    (void)pool.allocate_sync(SEPARATOR_SIZE);
  }
  for (auto const& fragment : fragments) {
    pool.deallocate_sync(fragment.pointer, fragment.size);
  }

  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    auto* ptr = pool.allocate_sync(REQUEST_SIZE);
    benchmark::DoNotOptimize(ptr);
    pool.deallocate_sync(ptr, REQUEST_SIZE);
  }
  state.SetItemsProcessed(state.iterations());
}

void BM_FragmentedBestFitAllocateSync(benchmark::State& state)
{ fragmented_best_fit_allocate_sync<rmm::mr::pool_memory_resource>(state); }

void BM_FragmentedBestFitIndexedAllocateSync(benchmark::State& state)
{ fragmented_best_fit_allocate_sync<rmm::mr::indexed_pool_memory_resource>(state); }

template <typename FreeList>
void populate_failed_best_fit_fixture(FreeList& blocks, std::size_t free_block_count)
{
  constexpr std::uintptr_t BASE{0x40000000};
  constexpr std::size_t BLOCK_STRIDE{4096};
  constexpr std::size_t BLOCK_SIZE{2048};

  // Descending insertion preserves the final address order while avoiding quadratic fixture setup.
  for (std::size_t i = free_block_count; i > 0; --i) {
    blocks.insert({reinterpret_cast<char*>(BASE + (i - 1) * BLOCK_STRIDE), BLOCK_SIZE, true});
  }
}

template <typename FreeList>
void repeated_warmed_failed_best_fit_lookup(benchmark::State& state)
{
  constexpr std::size_t REQUEST_SIZE{4096};
  FreeList blocks;
  populate_failed_best_fit_fixture(blocks, static_cast<std::size_t>(state.range(0)));

  // Besides checking the fixture, this lookup deliberately warms the indexed free list's
  // negative-result cache. Every timed iteration therefore measures a repeated warmed miss.
  auto const fixture_check = blocks.get_block(REQUEST_SIZE);
  if (fixture_check.is_valid()) {
    state.SkipWithError("failed-lookup fixture unexpectedly contains a fitting block");
    return;
  }

  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    auto const block = blocks.get_block(REQUEST_SIZE);
    benchmark::DoNotOptimize(block.pointer());
  }
  state.SetItemsProcessed(state.iterations());
}

void BM_RepeatedWarmedFailedBestFitLegacyFreeList(benchmark::State& state)
{ repeated_warmed_failed_best_fit_lookup<rmm::mr::detail::coalescing_free_list>(state); }

void BM_RepeatedWarmedFailedBestFitIndexedFreeList(benchmark::State& state)
{ repeated_warmed_failed_best_fit_lookup<rmm::mr::detail::indexed_coalescing_free_list>(state); }

constexpr benchmark::IterationCount FIRST_LOOKUP_ITERATIONS{1000};

template <typename FreeList>
void first_failed_best_fit_lookup(benchmark::State& state)
{
  constexpr std::size_t REQUEST_SIZE{4096};
  auto const free_block_count = static_cast<std::size_t>(state.range(0));

  // Validate the shape using a separate list so none of the measured fixtures has been queried.
  FreeList validation_fixture;
  populate_failed_best_fit_fixture(validation_fixture, free_block_count);
  if (validation_fixture.get_block(REQUEST_SIZE).is_valid()) {
    state.SkipWithError("failed-lookup fixture unexpectedly contains a fitting block");
    return;
  }

  std::vector<std::unique_ptr<FreeList>> fixtures;
  fixtures.reserve(static_cast<std::size_t>(FIRST_LOOKUP_ITERATIONS));
  for (benchmark::IterationCount i = 0; i < FIRST_LOOKUP_ITERATIONS; ++i) {
    auto blocks = std::make_unique<FreeList>();
    populate_failed_best_fit_fixture(*blocks, free_block_count);
    fixtures.push_back(std::move(blocks));
  }

  auto fixture = fixtures.cbegin();
  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    // Each fixture is consumed once, so both real and CPU time measure a genuinely first miss.
    auto const block = (*fixture++)->get_block(REQUEST_SIZE);
    benchmark::DoNotOptimize(block.pointer());
  }
  state.SetItemsProcessed(state.iterations());
}

void BM_FirstFailedBestFitLegacyFreeList(benchmark::State& state)
{ first_failed_best_fit_lookup<rmm::mr::detail::coalescing_free_list>(state); }

void BM_FirstFailedBestFitIndexedFreeList(benchmark::State& state)
{ first_failed_best_fit_lookup<rmm::mr::detail::indexed_coalescing_free_list>(state); }

template <typename PoolResource>
void cross_stream_best_fit(benchmark::State& state)
{
  auto const owner_count = static_cast<std::size_t>(state.range(0));
  constexpr std::size_t SMALL_SIZE{rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t REQUEST_SIZE{8 * rmm::CUDA_ALLOCATION_ALIGNMENT};
  constexpr std::size_t SEPARATOR_SIZE{rmm::CUDA_ALLOCATION_ALIGNMENT};
  auto const pool_size =
    (owner_count - 1) * (SMALL_SIZE + SEPARATOR_SIZE) + REQUEST_SIZE + SEPARATOR_SIZE;

  rmm::mr::simulated_memory_resource upstream{pool_size};
  PoolResource pool{upstream, pool_size, pool_size};
  rmm::device_async_resource_ref resource{pool};

  std::vector<rmm::cuda_stream> owners;
  owners.reserve(owner_count);
  std::vector<allocation> free_blocks;
  free_blocks.reserve(owner_count);
  for (std::size_t i = 0; i < owner_count; ++i) {
    owners.emplace_back();
    auto const size = (i + 1 == owner_count) ? REQUEST_SIZE : SMALL_SIZE;
    free_blocks.push_back(
      {resource.allocate(rmm::cuda_stream_legacy, size, rmm::CUDA_ALLOCATION_ALIGNMENT), size});
    (void)resource.allocate(
      rmm::cuda_stream_legacy, SEPARATOR_SIZE, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  for (std::size_t i = 0; i < owner_count; ++i) {
    resource.deallocate(owners[i].view(),
                        free_blocks[i].pointer,
                        free_blocks[i].size,
                        rmm::CUDA_ALLOCATION_ALIGNMENT);
  }

  rmm::cuda_stream requester;
  cudaEvent_t handoff{};
  RMM_CUDA_TRY(cudaEventCreateWithFlags(&handoff, cudaEventDisableTiming));

  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    auto* ptr = resource.allocate(requester.view(), REQUEST_SIZE, rmm::CUDA_ALLOCATION_ALIGNMENT);
    benchmark::DoNotOptimize(ptr);

    // Return the block to its donor owner without violating stream ordering so every iteration
    // exercises cross-stream selection rather than the same-stream fast path.
    RMM_CUDA_TRY(cudaEventRecord(handoff, requester.value()));
    RMM_CUDA_TRY(cudaStreamWaitEvent(owners.back().value(), handoff, 0));
    resource.deallocate(owners.back().view(), ptr, REQUEST_SIZE, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  owners.back().synchronize();
  requester.synchronize();
  RMM_CUDA_TRY(cudaEventDestroy(handoff));
  state.SetItemsProcessed(state.iterations());
}

void BM_CrossStreamBestFitLegacy(benchmark::State& state)
{ cross_stream_best_fit<rmm::mr::pool_memory_resource>(state); }

void BM_CrossStreamBestFitIndexed(benchmark::State& state)
{ cross_stream_best_fit<rmm::mr::indexed_pool_memory_resource>(state); }

template <typename PoolResource>
void selective_recovery_cycle(benchmark::State& state)
{
  auto const owner_count       = static_cast<std::size_t>(state.range(0));
  auto const total_block_count = static_cast<std::size_t>(state.range(1));
  auto const request_blocks    = static_cast<std::size_t>(state.range(2));
  constexpr std::size_t BLOCK_SIZE{rmm::CUDA_ALLOCATION_ALIGNMENT};
  auto const pool_size    = total_block_count * BLOCK_SIZE;
  auto const request_size = request_blocks * BLOCK_SIZE;

  rmm::mr::simulated_memory_resource upstream{pool_size};
  PoolResource pool{upstream, pool_size, pool_size};
  rmm::device_async_resource_ref resource{pool};
  std::vector<rmm::cuda_stream> owners(owner_count);
  rmm::cuda_stream requester;
  std::vector<void*> blocks(total_block_count);
  for (auto& block : blocks) {
    block = resource.allocate(rmm::cuda_stream_legacy, BLOCK_SIZE, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    resource.deallocate(
      owners[i % owner_count].view(), blocks[i], BLOCK_SIZE, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }

  std::vector<void*> rebuilt(total_block_count);
  for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
    auto* ptr = resource.allocate(requester.view(), request_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    benchmark::DoNotOptimize(ptr);

    // Recreate the alternating-owner fragmentation outside the timed region so the measurement is
    // the selective recovery allocation itself, not test-fixture reconstruction.
    state.PauseTiming();
    resource.deallocate(requester.view(), ptr, request_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    for (auto& block : rebuilt) {
      block = resource.allocate(requester.view(), BLOCK_SIZE, rmm::CUDA_ALLOCATION_ALIGNMENT);
    }
    for (std::size_t i = 0; i < rebuilt.size(); ++i) {
      resource.deallocate(
        owners[i % owner_count].view(), rebuilt[i], BLOCK_SIZE, rmm::CUDA_ALLOCATION_ALIGNMENT);
    }
    state.ResumeTiming();
  }

  for (auto& owner : owners) {
    owner.synchronize();
  }
  requester.synchronize();
  state.SetItemsProcessed(state.iterations());
}

void BM_WholeTreeRecovery(benchmark::State& state)
{ selective_recovery_cycle<rmm::mr::pool_memory_resource>(state); }

void BM_SelectiveRecovery(benchmark::State& state)
{ selective_recovery_cycle<rmm::mr::indexed_pool_memory_resource>(state); }

void recovery_shapes(benchmark::Benchmark* benchmark)
{
  for (auto const owners : {2, 4, 8, 16}) {
    for (auto const total_blocks : {64, 1024, 4096}) {
      benchmark->Args({owners, total_blocks, 16});
    }
  }
  benchmark->Iterations(50);
  benchmark->Unit(benchmark::kNanosecond);
}

void stream_owner_counts(benchmark::Benchmark* benchmark)
{
  for (auto const count : {1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 128}) {
    benchmark->Arg(count);
  }
  benchmark->Unit(benchmark::kNanosecond);
}

void free_block_depths(benchmark::Benchmark* benchmark)
{
  for (auto const depth : {16, 64, 256, 1024, 4096}) {
    benchmark->Arg(depth);
  }
  benchmark->Unit(benchmark::kNanosecond);
}

void below_index_free_block_depths(benchmark::Benchmark* benchmark)
{
  for (auto const depth : {16, 64, 256, 512, 1023}) {
    benchmark->Arg(depth);
  }
  benchmark->Unit(benchmark::kNanosecond);
}

void first_failed_lookup_depths(benchmark::Benchmark* benchmark)
{
  below_index_free_block_depths(benchmark);
  benchmark->Iterations(FIRST_LOOKUP_ITERATIONS);
}

}  // namespace

BENCHMARK(BM_FragmentedBestFitLegacyStream)->Apply(free_block_depths);
BENCHMARK(BM_FragmentedBestFitPerThreadStream)->Apply(free_block_depths);
BENCHMARK(BM_FragmentedBestFitIndexedLegacyStream)->Apply(free_block_depths);
BENCHMARK(BM_FragmentedBestFitIndexedPerThreadStream)->Apply(free_block_depths);
BENCHMARK(BM_FragmentedBestFitAllocateSync)->Apply(free_block_depths);
BENCHMARK(BM_FragmentedBestFitIndexedAllocateSync)->Apply(free_block_depths);
BENCHMARK(BM_RepeatedWarmedFailedBestFitLegacyFreeList)->Apply(below_index_free_block_depths);
BENCHMARK(BM_RepeatedWarmedFailedBestFitIndexedFreeList)->Apply(below_index_free_block_depths);
BENCHMARK(BM_FirstFailedBestFitLegacyFreeList)->Apply(first_failed_lookup_depths);
BENCHMARK(BM_FirstFailedBestFitIndexedFreeList)->Apply(first_failed_lookup_depths);
BENCHMARK(BM_CrossStreamBestFitLegacy)->Apply(stream_owner_counts);
BENCHMARK(BM_CrossStreamBestFitIndexed)->Apply(stream_owner_counts);
BENCHMARK(BM_WholeTreeRecovery)->Apply(recovery_shapes);
BENCHMARK(BM_SelectiveRecovery)->Apply(recovery_shapes);

BENCHMARK_MAIN();
