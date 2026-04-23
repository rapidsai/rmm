# Memory Resources

## Authoring custom memory resources

All memory resources — both those provided by RMM and those written by library authors against
RMM's resource concepts — must be safe to destroy at process shutdown.

A memory resource destructor may run in two very different contexts:

1. **During normal program execution**, where calling into the CUDA runtime is safe.
2. **After `main()` has returned**, as part of `exit()` / atexit handler processing. This
   happens whenever a memory resource is owned by a static or `thread_local` object, including
   the per-device resource map that RMM uses internally and any static container maintained by
   a consuming library. In this context, calling into the CUDA runtime or driver is
   **undefined behavior**: the primary context may already have been destroyed, and CUDA API
   calls may dereference released state and crash inside `libcuda` rather than returning an
   error.

To satisfy this contract, authors of memory resources (and of any destructor reachable from an
MR destructor, such as a `release()` helper) must follow one of these rules:

- **Do not call CUDA APIs from destructors at all.** This is the simplest option and should be
  preferred when feasible.

- **Consult {cpp:func}`rmm::process_is_exiting()`** in any destructor that would otherwise call
  a CUDA API. When `rmm::process_is_exiting()` returns `true`, skip the CUDA calls and allow
  the associated resources to leak; the operating system reclaims process memory, file
  descriptors, and driver-owned state when the process exits.

Calling `rmm::process_is_exiting()` is always safe from a destructor: it performs a single
atomic load and never calls into CUDA.

### Example

```cpp
class my_resource final : public rmm::mr::device_memory_resource {
 public:
  ~my_resource() override
  {
    if (!rmm::process_is_exiting()) {
      RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaFree(ptr_));
    }
  }
  ...
};
```

```{doxygengroup} memory_resources
:members:
:undoc-members:
:content-only:
```
