# Caching Memory Resource Design Notes

## Context

This document records the current state of the `rmm::mr::caching_memory_resource` prototype, the
parts of PyTorch's `CUDACachingAllocator` that informed the design, what has been implemented in
RMM so far, and what still needs investigation before this should be treated as production-ready.

The goal is not to port PyTorch's allocator wholesale. RMM has an existing memory-resource model,
existing stream-ordered resource infrastructure, and existing allocator components such as
`pool_memory_resource`, `arena_memory_resource`, `binning_memory_resource`, and
`fixed_size_memory_resource`. The intent of this work is to implement the PyTorch allocator policy
shape in an RMM-native resource with the smallest viable new surface area.

## PyTorch Allocator Behavior Studied

The relevant PyTorch implementation is in `~/rmm/pytorch/c10/cuda/CUDACachingAllocator.cpp` and
`~/rmm/pytorch/c10/cuda/CUDACachingAllocator.h`.

The allocator is built around these core ideas:

- Allocations are associated with streams.
- Freed blocks are reused preferentially on the same stream.
- Blocks can be split and coalesced.
- Small and large allocations use separate pools.
- Small requests are packed into larger upstream segments.
- Medium-large requests use a fixed large segment size to reduce fragmentation.
- Very large requests are rounded to a fixed granularity.
- On upstream allocation failure, cached free segments can be returned to CUDA and the allocation is retried.
- Some advanced behavior exists for CUDA Graph capture, expandable segments, tracing, snapshots, statistics, peer access, and IPC limitations.

The key size constants in PyTorch are:

- `kMinBlockSize = 512`
- `kSmallSize = 1 MiB`
- `kSmallBuffer = 2 MiB`
- `kLargeBuffer = 20 MiB`
- `kMinLargeAlloc = 10 MiB`
- `kRoundLarge = 2 MiB`

The key upstream allocation policy is equivalent to:

```cpp
if (size <= 1_MiB) {
  return 2_MiB;
} else if (size < 10_MiB) {
  return 20_MiB;
} else {
  return align_up(size, 2_MiB);
}
```

PyTorch's split policy is roughly:

- Small-pool blocks split when the remainder is at least 512 bytes.
- Expandable-segment blocks split when the remainder is at least 512 bytes.
- Large-pool blocks split only when the requested size is below `max_split_size` and the remainder is larger than 1 MiB.
- Oversized cached blocks have additional reuse constraints to avoid excessive fragmentation.

## RMM Components Reused

The current prototype reuses RMM's `stream_ordered_memory_resource` CRTP base and
`coalescing_free_list` block representation.

This gives the new resource:

- Thread-safe allocation and deallocation through the base resource mutex.
- Stream-event tracking.
- Same-stream reuse without synchronization.
- Reuse from other streams with event waits.
- Coalescing free lists.
- Best-fit search through `coalescing_free_list::get_block`.
- Compatibility with the RMM async memory-resource interface.

The prototype does not currently compose `pool_memory_resource`, `arena_memory_resource`, or
`binning_memory_resource` directly. A pure composition approach was considered but does not expose
enough of the required segment-level policy:

- The resource needs to decide upstream segment size before calling upstream.
- The resource needs to distinguish small-pool and large-pool split behavior.
- The resource needs to know which whole upstream segments can be released after allocation failure.
- The resource needs one coherent stream-ordered block model.

## Implemented Files

The prototype added these source files:

- `cpp/include/rmm/mr/caching_memory_resource.hpp`
- `cpp/include/rmm/mr/detail/caching_memory_resource_impl.hpp`
- `cpp/src/mr/caching_memory_resource.cpp`
- `cpp/src/mr/detail/caching_memory_resource_impl.cpp`
- `cpp/tests/mr/caching_mr_tests.cpp`

The build was wired through:

- `cpp/CMakeLists.txt`
- `cpp/tests/CMakeLists.txt`

Documentation was added to:

- `docs/cpp/memory_resources/memory_resources.md`

The benchmark harnesses were started but not completed:

- `cpp/benchmarks/random_allocations/random_allocations.cpp`
- `cpp/benchmarks/multi_stream_allocations/multi_stream_allocations_bench.cu`

## Current Public API

The current public resource is `rmm::mr::caching_memory_resource`.

Construction:

```cpp
explicit caching_memory_resource(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
  std::optional<std::size_t> max_split_size = std::nullopt);
```

Public helper methods:

- `get_upstream_resource()`
- `release()`
- `release_small_blocks()`
- `release_large_blocks()`
- `cached_small_bytes()`
- `cached_large_bytes()`
- `upstream_allocation_size(bytes)`

The helper methods were added to make early behavior testable and observable. Before stabilizing the
API, we should decide which of these belong in the long-term public interface and which should move
behind testing-only utilities or statistics adaptors.

## Implemented Behavior

### Upstream Segment Sizing

The prototype implements the PyTorch sizing policy:

- Requests of `0` bytes return `nullptr` through the inherited base behavior.
- Requests `<= 1 MiB` allocate `2 MiB` upstream segments.
- Requests `> 1 MiB` and `< 10 MiB` allocate `20 MiB` upstream segments.
- Requests `>= 10 MiB` allocate upstream sizes rounded up to `2 MiB`.

This is exposed by `upstream_allocation_size(bytes)` and covered by tests.

### Stream-Ordered Reuse

The resource inherits the stream ordering model from `stream_ordered_memory_resource`.

This means allocation uses a stream-associated event and free-list. Same-stream blocks can be reused
without waiting. Blocks freed on other streams can be reused after the allocating stream waits on the
other stream's event.

This has not yet received a dedicated cross-stream correctness test for `caching_memory_resource`.
It is inherited from existing RMM infrastructure, but because this resource has new block sizing and
release behavior, it still needs direct tests.

### Best-Fit Suballocation and Coalescing

The resource uses `coalescing_free_list`, so free blocks are searched using best fit and coalesced
when returned to the free list.

The prototype's allocation path is:

1. Align the requested size to RMM's CUDA allocation alignment in the base class.
2. Search the current stream's free list.
3. Search other stream free lists.
4. Merge and search other stream free lists if needed.
5. Grow from upstream using the caching allocator segment-size policy.
6. Split a block if the split policy permits it.

### Split Policy

The implemented split policy follows PyTorch's non-expandable segment policy:

- Small-pool blocks split when the remainder is at least 512 bytes.
- Large-pool blocks split only when the requested size is below `max_split_size` and the remainder
  is greater than 1 MiB.
- The default `max_split_size` is `std::numeric_limits<std::size_t>::max()`, matching PyTorch's
  default behavior when no `max_split_size_mb` setting is supplied.
- Oversized large-pool blocks are rejected for smaller requests using the same broad constraints as
  PyTorch: requests below `max_split_size` do not reuse blocks at or above `max_split_size`, and
  requests at or above `max_split_size` do not reuse blocks at least 20 MiB larger than requested.

Expandable-segment split behavior is not implemented.

### Cached Segment Release

The prototype tracks upstream segments separately from free-list blocks. On upstream allocation
failure during pool growth, it releases cached whole segments from the relevant pool and retries.

The current release logic follows PyTorch's native allocator invariant for non-expandable segments:
only whole, non-split free segments are returned to the upstream resource. The RMM implementation
tracks allocated blocks, synchronizes and merges all stream free lists during explicit release, then
removes an exact whole-segment free block before calling upstream deallocation.

This behavior is covered by tests for release after allocation failure, explicit release, live
segments, partially free segments, and cross-stream frees.

## Validation So Far

The new test target was built successfully:

```bash
SCCACHE_DISABLE=1 cmake --build cpp/build-caching-opencode -j2 --target CACHING_MR_TEST
```

The test binary was run directly:

```bash
./cpp/build-caching-opencode/gtests/CACHING_MR_TEST
```

Result:

- 15 tests passed.

Covered scenarios:

- Upstream segment sizing policy.
- Small allocation reuse.
- Cached segment release after forced upstream allocation failure.
- Explicit release of cached small-pool and large-pool bytes.
- No release while a segment contains a live allocation.
- No release while a segment is only partially free.
- Cross-stream free-list merge before release.
- Cross-stream allocation reuse.
- Deleted-stream allocation reuse.
- Destructor release of cached blocks.
- Small-pool no-split boundary behavior.
- Same-stream large-pool split behavior.
- Medium allocation behavior using 20 MiB upstream segments.
- Large-pool split behavior and `max_split_size` behavior.

The existing `cpp/build-opencode` tree was not reusable from `/home/coder/rmm` because its CMake
cache was created from a different absolute path. A fresh `cpp/build-caching-opencode` tree was used.
The first build attempt failed because `sccache` remote access returned HTTP 401, so validation was
rerun with `SCCACHE_DISABLE=1`.

## Known Correctness Gaps

### Remaining Test Gaps

The current focused tests cover the highest-risk release, split, stream reuse, and destructor cache
release behavior.

### Public API Needs Review

The introspection methods are useful for tests and benchmarking, but they may not all belong in the
stable public API.

Questions to answer:

- Should `cached_small_bytes()` and `cached_large_bytes()` be public?
- Should `upstream_allocation_size()` be public, or only documented as behavior?
- Should `release_small_blocks()` and `release_large_blocks()` be public, or should there only be `release()`?
- Should this resource expose statistics through `statistics_resource_adaptor` instead of bespoke counters?

### Destruction and Static Teardown Need Attention

RMM has known sensitivity around stateful memory resources during static teardown. The prototype's
destructor calls `release()`, and the base class also releases stream events and free lists.

This needs explicit validation against the static teardown patterns that affected
`pool_memory_resource`.

### Exception Safety Needs Review

The allocation failure path releases cached segments and retries. This path must be audited for:

- preserving CUDA error state behavior consistently with other RMM resources
- not swallowing non-OOM upstream failures
- not corrupting segment bookkeeping if retry throws
- correct behavior when upstream deallocation fails during release

### Alignment and Requested Size Semantics Need Review

The base resource aligns requests to `rmm::CUDA_ALLOCATION_ALIGNMENT`. The PyTorch allocator tracks
both rounded block size and originally requested size for statistics and snapshots.

The prototype currently only manages rounded allocation sizes. This is enough for allocation and
deallocation, but insufficient for PyTorch-like statistics or diagnostics.

## Features Not Yet Implemented

### CUDA Graph Private Pools

PyTorch has graph-private pools so memory addresses captured by CUDA Graphs remain valid across
graph replay. The current RMM prototype does not implement private graph pools.

This should be treated as a separate design effort. It likely needs an API for associating
allocations with a graph capture or user-provided pool identity.

### Expandable Segments

PyTorch optionally uses CUDA virtual memory APIs to reserve a large virtual address range and map
physical memory into it over time. This reduces fragmentation for workloads with slightly changing
allocation sizes.

The RMM prototype does not implement expandable segments. This would require new RMM components
around CUDA driver memory mapping APIs, peer access, IPC limitations, and unmap/remap policy.

### Statistics, Snapshots, and Tracing

PyTorch exposes detailed allocator state and trace history. The RMM prototype does not.

RMM already has `statistics_resource_adaptor` and `tracking_resource_adaptor`. We should first decide
how much can be achieved by composing those adaptors around `caching_memory_resource` before adding
resource-specific stats.

### Age-Based Garbage Collection

PyTorch can free older unused blocks when a garbage collection threshold is exceeded. The prototype
does not track block ages or implement threshold-based garbage collection.

This may be useful later, but it should come after exact segment accounting is implemented.

## Benchmarking Status

The benchmark harness work has started but was not completed in the interrupted session.

The following benchmark files were updated to include a `caching` resource option:

- `cpp/benchmarks/random_allocations/random_allocations.cpp`
- `cpp/benchmarks/multi_stream_allocations/multi_stream_allocations_bench.cu`

The intended comparison set is:

- `cuda_async_memory_resource`
- `pool_memory_resource`
- `caching_memory_resource`

The intended benchmark targets are:

- `RANDOM_ALLOCATIONS_BENCH`
- `MULTI_STREAM_ALLOCATIONS_BENCH`

The existing `cpp/build-opencode` tree was configured with `BUILD_BENCHMARKS=OFF`, so those targets
were unavailable. A separate benchmark build tree was started with:

```bash
cmake -S cpp -B cpp/build-bench-opencode -DBUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
```

That configure was interrupted before benchmark builds and runs completed.

Suggested next benchmark commands:

```bash
cmake -S cpp -B cpp/build-bench-opencode -DBUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build-bench-opencode -j2 --target RANDOM_ALLOCATIONS_BENCH MULTI_STREAM_ALLOCATIONS_BENCH
```

Example random-allocation runs:

```bash
./cpp/build-bench-opencode/gbenchmarks/RANDOM_ALLOCATIONS_BENCH \
  --resource=cuda_async \
  --numallocs=1000 \
  --maxsize=64 \
  --benchmark_min_time=0.2

./cpp/build-bench-opencode/gbenchmarks/RANDOM_ALLOCATIONS_BENCH \
  --resource=pool \
  --numallocs=1000 \
  --maxsize=64 \
  --benchmark_min_time=0.2

./cpp/build-bench-opencode/gbenchmarks/RANDOM_ALLOCATIONS_BENCH \
  --resource=caching \
  --numallocs=1000 \
  --maxsize=64 \
  --benchmark_min_time=0.2
```

Example multi-stream runs:

```bash
./cpp/build-bench-opencode/gbenchmarks/MULTI_STREAM_ALLOCATIONS_BENCH \
  --resource=cuda_async \
  --kernels=4 \
  --streams=4 \
  --warm=true \
  --benchmark_min_time=0.2

./cpp/build-bench-opencode/gbenchmarks/MULTI_STREAM_ALLOCATIONS_BENCH \
  --resource=pool \
  --kernels=4 \
  --streams=4 \
  --warm=true \
  --benchmark_min_time=0.2

./cpp/build-bench-opencode/gbenchmarks/MULTI_STREAM_ALLOCATIONS_BENCH \
  --resource=caching \
  --kernels=4 \
  --streams=4 \
  --warm=true \
  --benchmark_min_time=0.2
```

The benchmark command-line handling in these existing benchmarks should also be checked. In both
files, `--resource` is parsed by `cxxopts`, while Google Benchmark also consumes its own flags. The
benchmark declarations are filtered by the selected resource name. For side-by-side comparison in a
single process, it may be cleaner to add a dedicated mode or allow `--resource=all`.

## Recommended Next Implementation Work

### 1. Strengthen Tests

The deleted-stream reuse, destructor cache release, small-pool no-split boundary, and direct
same-stream large-pool split tests have been added.

### 2. Finish Benchmarks

Complete benchmark configuration with `BUILD_BENCHMARKS=ON`, build the benchmark targets, and record
results for at least:

- small allocation churn
- medium allocation churn
- mixed small/large allocation churn
- multi-stream temporary allocations
- warmed and cold cache cases

### 3. Decide Public API Shape

Before merging broadly, decide whether the public helper methods are stable API or test aids. Remove
or hide anything that is not needed by users.

### 4. Revisit Composition with Existing Resources

Revisit whether any logic should move into reusable components:

- a segment-size policy helper
- a PyTorch-style split policy helper
- protected base-class hooks for release-capable stream-ordered resources

This should happen only after the behavior is correct; premature abstraction would make the current
prototype harder to reason about.

## Open Design Questions

- Should `caching_memory_resource` use `cuda_memory_resource` or `cuda_async_memory_resource` as the recommended upstream default?
- Should the constructor accept a memory limit similar to PyTorch's memory fraction behavior?
- Should cached whole segments be released only on OOM, or should there be threshold-based proactive trimming?
- Should small and large segment sizes be configurable?
- Should this resource support graph-private pools, or should graph capture be handled by a separate adaptor/resource?
- Should expandable segments be a mode of this resource or a separate upstream resource?

## Current Recommendation

The current prototype now has the core non-expandable PyTorch-style segment sizing, split policy,
stream-ordered reuse, and whole-segment release behavior needed for further evaluation.

The next step is broader validation: deleted-stream/destructor tests, API review, and benchmark
results for small, medium, mixed, and multi-stream workloads.
