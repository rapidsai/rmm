# Caching Allocator Plan

## Goal

Make `rmm::mr::caching_memory_resource` correct and reviewable as an RMM-native allocator inspired by PyTorch's `CUDACachingAllocator`, without accidentally implying full PyTorch behavioral compatibility.

## Decisions To Make

### 1. Compatibility Target

Question: Should this allocator be PyTorch-compatible where possible, or PyTorch-inspired but RMM-native?

Decision: Treat it as PyTorch-inspired and RMM-native. RMM already has stream-ordered memory-resource semantics, and forcing exact PyTorch stream behavior would fight that architecture.

Consequence: Document intentional divergences clearly, especially cross-stream reuse.

### 2. Stream Reuse Semantics

Question: Should freed blocks be reusable across streams through RMM's event-wait machinery, or should reuse be restricted to the original allocation stream like PyTorch?

Decision: Preserve the ability to test both policies. Performance should determine whether the default allows RMM stream-ordered cross-stream reuse or uses PyTorch-style same-stream reuse.

Implementation shape: Add constructor options on `caching_memory_resource`.

Default: Match PyTorch behavior where feasible. For stream reuse, this means the default should be same-stream-oriented unless RMM's existing stream-ordered model makes exact PyTorch behavior impractical.

Required work:

- Document this as a configurable policy, not accidental behavior.
- Add tests for cross-stream reuse policy.
- Add tests for same-stream-only policy.
- Add benchmarks that compare cross-stream and same-stream-only reuse on single-stream, multi-stream, warm-cache, and cold-cache workloads.

### 3. Small And Large Pool Separation

Question: Must small and large pools be physically separate free lists, or is segment-origin metadata sufficient?

Decision: Preserve the ability to test both policies. Performance and fragmentation results should determine whether the default uses structurally separate pools or segment-origin filtering.

Implementation shape: Add constructor options on `caching_memory_resource`. Prefer policy plumbing that can select the current segment-origin filtering behavior and a PyTorch-style separated-pool behavior.

Default: Match PyTorch behavior where feasible. For pool handling, this means structurally separate small and large allocation pools should be the default target unless benchmark results or RMM architecture show a clear reason not to.

Required work:

- Add tests that expose the intended behavior first.
- Add benchmarks that compare structurally separate pools against segment-origin filtering.
- If separate pools are required, change the block storage model deliberately rather than adding more predicates.

### 4. Invalid Free Handling

Question: Should invalid pointer or double-free behavior be checked in release builds, debug builds, or left undefined like some allocator APIs?

Decision: Deallocation must remain `noexcept`; invalid pointer and double-free behavior is undefined behavior.

Future option: Add compile-time allocation tracking, for example `RMM_POOL_TRACK_ALLOCATIONS`, with debug assertions if needed.

Required work:

- Do not add runtime throwing behavior to deallocation.
- Do not make safety tracking part of the default implementation.
- If tracking is added later, guard it behind a compile-time option and test only that option.

### 5. Request Rounding

Question: Should RMM implement PyTorch's 512-byte minimum rounding and optional power-of-two divisions, or keep `CUDA_ALLOCATION_ALIGNMENT` rounding?

Recommendation: Keep RMM alignment for now. Add PyTorch-style configurable rounding only if benchmark results show fragmentation problems or compatibility requires it.

Required work:

- Document the divergence.
- Add boundary tests around sizes below 512 bytes and around split thresholds.

### 6. OOM Release Policy

Question: Should OOM retry release only same-pool whole segments, or should it match PyTorch's broader release sequence?

Recommendation: Keep same-pool whole-segment release initially. Expand only after pool separation and invalid-free behavior are settled.

Required work:

- Add explicit tests for same-pool release.
- Add explicit tests that opposite-pool cached blocks are not released, if this remains the policy.

### 7. Public API Shape

Question: Are `cached_small_bytes()`, `cached_large_bytes()`, `upstream_allocation_size()`, `release_small_blocks()`, and `release_large_blocks()` stable public API or test scaffolding?

Decision: Keep the helper methods public for now. They can be refactored later after the allocator behavior and benchmark needs are clearer.

Required work:

- Prefer testing via behavior and upstream limiter state where possible.
- Avoid adding more public helper APIs until there is a concrete user need.

## Proposed Implementation Order

### Phase 1: Lock Down Intentional Semantics

1. Add design documentation for intentional divergences from PyTorch.
2. Add policy tests for stream reuse: cross-stream reuse enabled and same-stream-only behavior.
3. Add policy tests for pool handling: segment-origin filtering and structurally separate pools.
4. Add tests for request-size and split-boundary behavior.

Exit criteria:

- We can state exactly which PyTorch semantics are intentionally not implemented.
- Tests fail if small/large pool behavior drifts unexpectedly.
- Benchmarks can compare the stream and pool policy choices without changing test expectations.

### Phase 2: Add Performance Gates

1. Finish benchmark support for selecting allocator policy.
2. Compare stream reuse policies on multi-stream churn workloads.
3. Compare pool representation policies on mixed small/large allocation workloads.
4. Generate Markdown benchmark reports but do not commit them until explicitly requested.

Exit criteria:

- The default stream and pool policies are selected by measured behavior, not assumption.
- Non-default policy remains available if it has plausible workload value.

### Phase 3: Revisit Pool Representation

1. If pool-isolation tests show current predicate filtering is insufficient, split free-list storage into small and large pools.
2. Keep the diff minimal and avoid rewriting stream-ordering machinery unless required.
3. Re-run caching tests and add one fragmentation-focused test.

Exit criteria:

- Small and large segment policies are enforced structurally, not accidentally.

### Phase 4: Expand OOM Behavior Deliberately

1. Decide whether opposite-pool cached segments should be releasable on OOM.
2. Decide whether oversize cached segments need PyTorch-like preferential release.
3. Add tests before implementation.

Exit criteria:

- OOM retry behavior is explicit, covered, and documented.

### Phase 5: Defer Larger Features

Defer these until the core allocator is stable:

- CUDA graph private pools
- expandable segments
- memory snapshots and tracing
- PyTorch-style runtime configuration parsing
- garbage-collection threshold

## Immediate Next Step

Start with pool policy first. Add separate enum constructor options for stream reuse and pool handling, but implement the pool policy before changing stream reuse behavior.

Initial enum shape:

- `enum class stream_reuse_policy { same_stream, cross_stream };`
- `enum class pool_policy { separate, unified };`

Default target:

- `stream_reuse_policy::same_stream`, matching PyTorch where feasible.
- `pool_policy::separate`, matching PyTorch where feasible.

Compatibility option:

- Preserve current behavior with `stream_reuse_policy::cross_stream` and `pool_policy::unified` if the implementation can support it without excessive complexity.

Benchmark report location:

- Generate Markdown benchmark reports under `caching-allocator-reports/` at the repository root.
- Do not commit benchmark reports unless explicitly requested.

## Open Questions

1. Which OOM fallback subset should RMM implement first after pool separation?

## Decisions Made

- Compatibility target: PyTorch-inspired but RMM-native.
- Policy selection: constructor options, not separate classes.
- Constructor option shape: separate enums.
- Implementation order: pool policy first, then stream policy.
- Defaults: PyTorch behavior where feasible.
- Benchmark priorities: multi-stream churn and mixed allocation patterns.
- Benchmark reports: generate Markdown reports in `caching-allocator-reports/`, but do not commit them until explicitly requested.
- Benchmark report filenames should include workload, policy, git commit, CUDA device, and timestamp.
- Separate pool OOM fallback: match PyTorch where feasible; likely allow OOM fallback to release across pools.
- `pool_policy::unified`: public for now.
- Stream policy implementation: avoid changing `stream_ordered_memory_resource`; implement any same-stream behavior in `caching_memory_resource_impl` or adjacent caching-specific code.
- Invalid deallocation: deallocation remains `noexcept`; invalid pointer and double-free behavior is undefined behavior. Optional debug tracking can be considered later behind a compile-time option.
- Public helper API: leave existing helpers public for now.

## Pool-Policy-First Work Breakdown

### Step 1: Add Public Enum Plumbing

- Add `pool_policy` and `stream_reuse_policy` enums to the public caching allocator header.
- Add constructor parameters with defaults matching the target behavior.
- Thread those values into `caching_memory_resource_impl`.
- Do not change behavior yet unless required to compile cleanly.

### Step 2: Add Pool Policy Tests

- Add tests that differentiate separate pools from unified pools.
- Separate-pool test: a small request should not consume a large-pool free block.
- Separate-pool test: a large request should not consume a small-pool free block.
- Unified-pool test: preserve current behavior where a fitting block can be reused across pool categories, if that remains supported.

### Step 3: Implement Separate Pool Policy

Status: Implemented with `caching_memory_resource_pool_policy`.

- Prefer the smallest structural change that keeps stream-ordering behavior intact.
- Avoid large refactors of `stream_ordered_memory_resource` unless the existing free-list model cannot express pool separation cleanly.
- Keep `pool_policy::unified` available for benchmarks if the extra complexity is modest.

### Step 4: Benchmark Mixed Allocations

Status: Benchmark binaries accept caching pool and stream policy options. Report generation is available through `cpp/benchmarks/caching_allocator_reports.py`.

- Build mixed small/large allocation benchmark coverage.
- Compare `pool_policy::separate` and `pool_policy::unified`.
- Write Markdown reports under `caching-allocator-reports/`.

### Step 5: Revisit Stream Policy

Status: Implemented with `caching_memory_resource_stream_reuse_policy`.

- After pool behavior is explicit, add stream policy tests and implementation.
- Benchmark multi-stream churn for `stream_reuse_policy::same_stream` versus `stream_reuse_policy::cross_stream`.

Benchmark reports are generated but intentionally ignored by git.

## OOM Fallback Options

PyTorch's native allocator has a staged OOM path. It does not simply release one pool and retry. The relevant simplified sequence is:

1. Try allocator callbacks.
2. Try garbage collection if configured.
3. Try the upstream allocation.
4. Release cached oversize blocks that could satisfy the request, then retry.
5. Release all non-split cached blocks, then retry.
6. Report OOM.

RMM does not need to implement all of this immediately. The useful implementation choices are below.

### Option A: Same-Pool Whole-Segment Release Only

Behavior: On upstream allocation failure, release cached whole segments from the same pool as the request, then retry once.

Pros:

- This is closest to the current implementation.
- Small diff and easy to reason about.
- Avoids surprising cross-pool cache eviction.

Cons:

- Not PyTorch-like when memory is stranded in the opposite pool.
- Can fail an allocation even though enough releasable cached memory exists globally.
- Weak benchmark comparator for PyTorch-inspired behavior.

Use if: We prioritize minimal allocator complexity over PyTorch-like OOM recovery.

### Option B: Release All Whole Segments Across Pools

Behavior: On upstream allocation failure, release all cached whole segments from both small and large pools, then retry once.

Pros:

- Simple and robust.
- Closer to PyTorch's final `release_cached_blocks` stage.
- Handles memory stranded in the opposite pool.
- Easy to test with `limiting_resource_adaptor`.

Cons:

- More aggressive than PyTorch's staged path because it skips targeted oversize release.
- May hurt performance by flushing useful cached segments unnecessarily.
- Blurs small/large pool isolation during OOM recovery.

Use if: We want a simple PyTorch-inspired recovery path before implementing finer-grained release.

### Option C: Targeted Oversize Whole-Segment Release, Then All Whole Segments

Behavior: On upstream allocation failure, first release cached whole segments that are oversized relative to the failed request and likely to reduce fragmentation, then retry. If that fails, release all whole segments and retry again.

Pros:

- Closest practical subset of PyTorch's native path without implementing full stats, GC, and callbacks.
- Avoids flushing the entire cache when a targeted release is enough.
- Better benchmark candidate for mixed allocation workloads.

Cons:

- More complex release selection and tests.
- Requires carefully defining `oversize` in RMM terms.
- More chances to diverge subtly from PyTorch while appearing compatible.

Use if: Benchmark evidence shows Option B is too disruptive and we need a more selective release path.

### Option D: Full PyTorch-Like OOM Pipeline

Behavior: Implement callbacks, garbage collection threshold, upstream retry staging, oversize release, all non-split cached release, detailed OOM accounting, and diagnostics.

Pros:

- Best PyTorch behavioral match.
- Enables future statistics and diagnostics work.

Cons:

- Too large for the current prototype.
- Adds features RMM may not need or may prefer to express through existing adaptors.
- High review and maintenance cost.

Use if: We explicitly decide this resource should become a near-port of PyTorch's allocator policy.

Implementation status: Options B and C are implemented behind `caching_memory_resource_oom_fallback_policy`.

Default: `release_oversized_then_all`, because it is the closer PyTorch-inspired staged fallback.
