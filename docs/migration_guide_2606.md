# RMM 26.06 Migration Guide

RMM 26.06 completes the migration to CCCL-native memory resources. This release
removes `device_memory_resource`, eliminates the `Upstream` template parameter
from all resources and adaptors, introduces `cuda::mr::shared_resource`-based
ownership, and removes several legacy APIs. This guide covers the breaking
changes and shows how to update consuming code.

## Summary of Breaking Changes

RMM's memory resource model changes from inheritance-based resources to
CCCL-native value resources:

- `device_memory_resource` is removed; resources must now satisfy CCCL's
  `cuda::mr::resource` concept directly.
- Resources and adaptors are no longer templated on `Upstream`.
- Upstream resources are stored as
  `cuda::mr::any_resource<cuda::mr::device_accessible>` instead of raw pointers.
- Resources with non-trivial state use `cuda::mr::shared_resource<Impl>` for
  internal shared ownership.

Several APIs change as a consequence:

- Allocation order changes from `allocate(size, stream)` to
  `allocate(stream, size, alignment)`. This aligns with the CCCL memory
  resource design.
- Pointer-based per-device resource setters are replaced by `any_resource`
  setters.
- `owning_wrapper` is removed.
- Resource implementations are compiled into `librmm` instead of being fully
  header-only.

## Migration Checklist

### RMM library changes

C++:

- {ref}`Replace pointer-based per-device resource setters <rmm-2604-2606-per-device-resource>`.
- {ref}`Update device_buffer resource arguments <rmm-2604-2606-device-buffer>`.
- {ref}`Update resource_ref aliases and pointer conversions <rmm-2604-2606-resource-ref-aliases>`.
- {ref}`Link against librmm <rmm-2604-2606-compiled-resources>`.

### Memory resource changes

C++:

- {ref}`Remove device_memory_resource includes and inheritance <rmm-2604-2606-device-memory-resource>`.
- {ref}`Remove Upstream template parameters <rmm-2604-2606-template-parameters>`.
- {ref}`Pass upstream resources by value <rmm-2604-2606-upstream-any-resource>`.
- {ref}`Update allocate/deallocate calls <rmm-2604-2606-allocation-signatures>`.
- {ref}`Remove owning_wrapper usage <rmm-2604-2606-owning-wrapper>`.
- {ref}`Replace do_allocate/do_deallocate overrides <rmm-2604-2606-custom-resource-guide>`.
- {ref}`Add allocate_sync/deallocate_sync <rmm-2604-2606-custom-resource-guide>`.
- {ref}`Ensure custom resources are copyable <rmm-2604-2606-copyable-custom-resources>`.
- {ref}`Dereference resource pointers for ref APIs <rmm-2604-2606-resource-pointers>`.
- {ref}`Use resource_cast instead of dynamic_cast <rmm-2604-2606-resource-cast>`.
- {ref}`Replace cuda::stream_ref{} <rmm-2604-2606-stream-ref-default>`.

Cython:

- {ref}`Update Cython .pxd resource declarations <rmm-2604-2606-cython-pxd>`.
- {ref}`Update downstream Cython function signatures <rmm-2604-2606-cython-downstream-pxd>`.
- {ref}`Update custom Cython resource wrappers <rmm-2604-2606-cython-resource-ref-helpers>`.

## RMM Library Changes

### C++

(rmm-2604-2606-per-device-resource)=
### Per-Device Resource API Changes

The pointer-based per-device resource setter functions are removed. Use the
owning `any_resource` setters instead. The `_ref` setters and reset functions
still exist in 26.06 as deprecated compatibility APIs, but will be removed in
26.08:

Replacements:

- `get_current_device_resource()` returning `device_memory_resource*` →
  `get_current_device_resource_ref()` returning `device_async_resource_ref`.
- `set_current_device_resource(device_memory_resource*)` →
  `set_current_device_resource(cuda::mr::any_resource<cuda::mr::device_accessible>)`.
- `get_per_device_resource(cuda_device_id)` returning `device_memory_resource*` →
  `get_per_device_resource_ref(cuda_device_id)` returning `device_async_resource_ref`.
- `set_per_device_resource(cuda_device_id, device_memory_resource*)` →
  `set_per_device_resource(cuda_device_id, cuda::mr::any_resource<cuda::mr::device_accessible>)`.
- `reset_per_device_resource_ref(cuda_device_id)` → `reset_per_device_resource(cuda_device_id)`.
- `reset_current_device_resource_ref()` → `reset_current_device_resource()`.

The `set_*` functions now return `cuda::mr::any_resource<cuda::mr::device_accessible>`,
an owning type-erased resource that holds the previously set resource. This can
be used for RAII-style scoped resource replacement.

```cpp
// 26.04
rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool{&cuda_mr, size};
rmm::mr::set_current_device_resource(&pool);
auto* mr = rmm::mr::get_current_device_resource();

// 26.06
rmm::mr::pool_memory_resource pool{cuda_mr, size};
auto prev = rmm::mr::set_current_device_resource(pool);
auto ref = rmm::mr::get_current_device_resource_ref();
// When done, restore: rmm::mr::set_current_device_resource(std::move(prev));
// Or reset to default: rmm::mr::reset_current_device_resource();
```


(rmm-2604-2606-device-buffer)=
### `device_buffer` Internal Changes

`rmm::device_buffer` now stores a `cuda::mr::any_resource<cuda::mr::device_accessible>`
internally instead of a raw `device_memory_resource*`. The public API is largely
the same, but:

- Constructors accept `cuda::mr::any_resource<cuda::mr::device_accessible>`.
- `memory_resource()` returns `rmm::device_async_resource_ref` (was
  `device_memory_resource*` in older releases).

(rmm-2604-2606-resource-ref-aliases)=
### `resource_ref` Type Aliases Updated

The `rmm::device_async_resource_ref` and related type aliases in
`rmm/resource_ref.hpp` are now wrappers around CCCL's `cuda::mr::resource_ref`
(async) and `cuda::mr::synchronous_resource_ref` respectively, with added
compatibility for `shared_resource`-derived types.

These types are implicitly constructible from any resource satisfying the
corresponding CCCL concept, and from `cuda::mr::any_resource`.

Since `device_memory_resource` no longer exists, `device_async_resource_ref`
can no longer be constructed from a `device_memory_resource*`. Code that
constructed a `resource_ref` from a `device_memory_resource` pointer must
be updated to pass the concrete resource type directly.

(rmm-2604-2606-compiled-resources)=
### Resource Implementations Are Compiled, Not Header-Only

In 26.04, RMM resource implementations were entirely header-only (template
definitions lived in the headers). In 26.06, many resources and adaptors have
their implementations in compiled translation units under `cpp/src/mr/`. The
public headers contain only the class declaration and inline accessors.

This means downstream projects that use RMM memory resources **must link against
`librmm`**. If your project previously relied on RMM being header-only for the
memory resource layer, you will need to update your build system to link the
`rmm` library target.

For CMake users, this should already work if you use `find_package(rmm)` or
`rapids_find_package(rmm)` — the imported `rmm::rmm` target handles linking.

## Memory Resource Changes

### C++

(rmm-2604-2606-device-memory-resource)=
### `device_memory_resource` Base Class Is Removed

**Header removed:** `rmm/mr/device_memory_resource.hpp`

In 26.04, all RMM memory resources — both base resources like
`cuda_memory_resource` and adaptors like `pool_memory_resource` — inherited from
`rmm::mr::device_memory_resource` and overrode `do_allocate` / `do_deallocate`
virtual methods. In 26.06, this base class no longer exists. Every resource
(base and adaptor alike) instead satisfies CCCL's `cuda::mr::resource` concept
directly.

This means code that relied on polymorphism through `device_memory_resource*`
(e.g., storing heterogeneous resources in a container of `device_memory_resource*`)
must switch to `device_async_resource_ref` or `cuda::mr::any_resource<cuda::mr::device_accessible>`.

**What to change:**
- Remove any `#include <rmm/mr/device_memory_resource.hpp>`.
- Remove any inheritance from `device_memory_resource` in custom resources.
- Remove `do_allocate` / `do_deallocate` overrides; implement `allocate` and
  `deallocate` with the CCCL signature instead (see
  {ref}`Allocation Signature Changes <rmm-2604-2606-allocation-signatures>`).
- Remove calls to `device_memory_resource::allocate(bytes, stream)`. Use
  `device_async_resource_ref` or call the resource's `allocate(stream, bytes)` directly.
- Replace `device_memory_resource*` variables with `device_async_resource_ref`
  (non-owning) or `cuda::mr::any_resource<cuda::mr::device_accessible>` (owning).

```cpp
// 26.04
#include <rmm/mr/device_memory_resource.hpp>

class my_resource : public rmm::mr::device_memory_resource {
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override { ... }
  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) override { ... }
};

// 26.06
#include <cuda/memory_resource>

class my_resource {
 public:
  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment = 256) { ... }
  void deallocate(cuda::stream_ref stream, void* ptr, std::size_t bytes,
                  std::size_t alignment = 256) noexcept { ... }
  bool operator==(my_resource const&) const noexcept { ... }
  bool operator!=(my_resource const& other) const noexcept { return !(*this == other); }

  // Declare properties via ADL friend:
  constexpr friend void get_property(my_resource const&, cuda::mr::device_accessible) noexcept {}
};
static_assert(cuda::mr::resource_with<my_resource, cuda::mr::device_accessible>);
```

(rmm-2604-2606-template-parameters)=
### Template Parameters Removed from All Resources and Adaptors

Every resource and adaptor that previously took an `Upstream` template parameter
is now a non-template class. The upstream is accepted as a
`cuda::mr::any_resource<cuda::mr::device_accessible>` instead of a typed pointer.

Remove the `Upstream` template parameter from these type names:

- `pool_memory_resource<Upstream>` → `pool_memory_resource`
- `arena_memory_resource<Upstream>` → `arena_memory_resource`
- `fixed_size_memory_resource<Upstream>` → `fixed_size_memory_resource`
- `binning_memory_resource<Upstream>` → `binning_memory_resource`
- `logging_resource_adaptor<Upstream>` → `logging_resource_adaptor`
- `tracking_resource_adaptor<Upstream>` → `tracking_resource_adaptor`
- `statistics_resource_adaptor<Upstream>` → `statistics_resource_adaptor`
- `aligned_resource_adaptor<Upstream>` → `aligned_resource_adaptor`
- `limiting_resource_adaptor<Upstream>` → `limiting_resource_adaptor`
- `thread_safe_resource_adaptor<Upstream>` → `thread_safe_resource_adaptor`
- `prefetch_resource_adaptor<Upstream>` → `prefetch_resource_adaptor`
- `failure_callback_resource_adaptor<Upstream, ExceptionType>` →
  `failure_callback_resource_adaptor<ExceptionType>`

```cpp
// 26.04
rmm::mr::cuda_memory_resource cuda_mr;
rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool{&cuda_mr, initial_size};
rmm::mr::logging_resource_adaptor<decltype(pool)> logged{&pool, "log.csv"};

// 26.06
rmm::mr::cuda_memory_resource cuda_mr;
rmm::mr::pool_memory_resource pool{cuda_mr, initial_size};
rmm::mr::logging_resource_adaptor logged{pool, "log.csv"};
```

(rmm-2604-2606-upstream-any-resource)=
### Upstream Resources Are Passed by `any_resource` Value, Not Pointer

Constructors for adaptors and pool-based resources now take
`cuda::mr::any_resource<cuda::mr::device_accessible>` (an owning type-erased
resource) instead of a raw pointer to the upstream.

A `cuda::mr::any_resource<cuda::mr::device_accessible>` is constructible from
any object that satisfies `cuda::mr::resource_with<T, cuda::mr::device_accessible>`,
so you can pass a resource object directly.

```cpp
// 26.04 - pass a device_memory_resource pointer
rmm::mr::cuda_memory_resource cuda_mr;
rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool{&cuda_mr, pool_size};

// 26.06 — pass the resource directly
rmm::mr::cuda_memory_resource cuda_mr;
rmm::mr::pool_memory_resource pool{cuda_mr, pool_size};
```

(rmm-2604-2606-allocation-signatures)=
### Allocation and Deallocation Signature Changes

The CCCL resource concept uses a different argument order and adds an alignment
parameter:

```cpp
// 26.04 (device_memory_resource virtual interface)
void* allocate(std::size_t bytes, cuda_stream_view stream);
void  deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream);

// 26.06 (CCCL resource concept)
void* allocate(cuda::stream_ref stream, std::size_t bytes,
               std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);
void  deallocate(cuda::stream_ref stream, void* ptr, std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;
```

Key differences:
- **Stream is the first parameter** (was second for `allocate`, third for `deallocate`).
- **`alignment` parameter** is new (has a default value).
- **`cuda::stream_ref`** replaces `rmm::cuda_stream_view` in the resource interface.
  `rmm::cuda_stream_view` is implicitly convertible to `cuda::stream_ref`.
- **`deallocate` is `noexcept`**.

Resources also provide synchronous methods `allocate_sync(bytes, alignment)` and
`deallocate_sync(ptr, bytes, alignment)`.

If you call allocation methods through `device_async_resource_ref`, the
wrapper preserves the `allocate(stream, bytes)` calling convention.

### Resources Use `cuda::mr::shared_resource` for Ownership

All RMM resources with non-trivial state now inherit from
`cuda::mr::shared_resource<detail::*_impl>`. This includes pool resources,
all adaptors, and also base resources like `cuda_async_memory_resource` and
`callback_memory_resource`. Only truly trivial resources like
`cuda_memory_resource` remain standalone. This means:

- **Resources are copyable** and copies share ownership of the underlying state.
- Resource lifetime is managed by reference counting internally; no need
  for `std::shared_ptr` or `std::unique_ptr` wrappers externally.
- Each resource class is a thin public API shell; implementation details
  live in a `detail::*_impl` class.

```cpp
// 26.06 — copying a pool shares the same underlying pool
rmm::mr::pool_memory_resource pool{cuda_mr, pool_size};
auto pool_copy = pool;  // both reference the same pool state
```

(rmm-2604-2606-owning-wrapper)=
### `owning_wrapper` Is Removed

**Header removed:** `rmm/mr/owning_wrapper.hpp`

`owning_wrapper` was used to keep an upstream resource alive when APIs stored
resources as `rmm::mr::device_memory_resource*`. In 26.06, those owning APIs
store `cuda::mr::any_resource<cuda::mr::device_accessible>` by value instead.
The `any_resource` holds the concrete resource with internal shared ownership,
keeping it alive while minimizing the cost of copying resource handles.

```cpp
// 26.04
auto cuda_mr = std::make_shared<rmm::mr::cuda_memory_resource>();
auto pool = rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
    cuda_mr, pool_size);

// 26.06 — resources are value types with shared ownership
rmm::mr::cuda_memory_resource cuda_mr;
rmm::mr::pool_memory_resource pool{cuda_mr, pool_size};
// pool is stored by value; its state is kept alive by internal shared ownership
```

(rmm-2604-2606-custom-resource-guide)=
### Custom Resource Implementation Guide

If you maintain a custom memory resource, here is the minimal migration:

```cpp
// 26.04
class my_resource : public rmm::mr::device_memory_resource {
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override {
    return upstream_->allocate(bytes, stream);
  }
  void do_deallocate(void* p, std::size_t bytes, rmm::cuda_stream_view stream) override {
    upstream_->deallocate(p, bytes, stream);
  }
  bool do_is_equal(device_memory_resource const& other) const noexcept override {
    return this == &other;
  }
  rmm::mr::device_memory_resource* upstream_;
};

// 26.06
class my_resource {
 public:
  explicit my_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream)
    : upstream_(std::move(upstream)) {}

  // Async (stream-ordered) interface
  void* allocate(cuda::stream_ref stream, std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
    return upstream_.allocate(stream, bytes, alignment);
  }
  void deallocate(cuda::stream_ref stream, void* p, std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
    upstream_.deallocate(stream, p, bytes, alignment);
  }

  // Synchronous interface — required by the CCCL resource concept
  void* allocate_sync(std::size_t bytes,
                      std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
    auto* ptr = allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
    RMM_CUDA_TRY(cudaStreamSynchronize(cudaStream_t{nullptr}));
    return ptr;
  }
  void deallocate_sync(void* p, std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, p, bytes, alignment);
  }

  bool operator==(my_resource const& other) const noexcept {
    return this == &other;
  }
  bool operator!=(my_resource const& other) const noexcept {
    return !(*this == other);
  }

  constexpr friend void get_property(my_resource const&, cuda::mr::device_accessible) noexcept {}

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
};
static_assert(cuda::mr::resource_with<my_resource, cuda::mr::device_accessible>);
```

RMM adaptors store upstream resources as
`cuda::mr::any_resource<cuda::mr::device_accessible>` so the upstream resource
stays alive with the adaptor state.

> **Note:** The CCCL `resource` concept requires both the async interface
> (`allocate`/`deallocate` with `stream_ref`) and the synchronous interface
> (`allocate_sync`/`deallocate_sync` without a stream). If you omit the
> synchronous methods, `static_assert` will fail. A simple implementation can
> delegate to the async methods with a null stream as shown above, synchronizing
> before returning from allocation.

(rmm-2604-2606-copyable-custom-resources)=
### Custom Resources Must Be Copyable for `any_resource`

`cuda::mr::any_resource<cuda::mr::device_accessible>` (used internally by
`device_buffer`, per-device resource storage, and as a common owning
type-erased resource) requires that the stored type is both **copyable and
movable**. If your custom resource holds non-copyable members like
`std::shared_mutex`, `std::unique_ptr`, or `std::unordered_set`, it cannot
be stored directly in `any_resource`.

A lightweight fix for small custom resources is to wrap shared state in
`std::shared_ptr`:

```cpp
// Won't work — shared_mutex is not copyable
class my_tracking_resource {
  std::shared_mutex mutex_;                // NOT copyable
  std::unordered_set<void*> allocations_;  // copyable, but expensive
  // ...
};

// Fix: wrap shared state in shared_ptr
class my_tracking_resource {
  struct shared_state {
    std::shared_mutex mutex;
    std::unordered_set<void*> allocations;
  };
  std::shared_ptr<shared_state> state_{std::make_shared<shared_state>()};
  // Now the resource is copyable; copies share the tracking state
  // ...
};
```

RMM's adaptor implementations use a separate implementation class held through
`cuda::mr::shared_resource`. See `rmm::mr::limiting_resource_adaptor` and
`rmm::mr::detail::limiting_resource_adaptor_impl` for an example of that
structure.

(rmm-2604-2606-resource-pointers)=
### Resource Pointers No Longer Convert to `device_async_resource_ref`

In 26.04, `device_async_resource_ref` was implicitly constructible from any
`device_memory_resource*`, so code could pass a raw pointer wherever a ref was
expected. In 26.06, `device_memory_resource` is gone, so pointers to concrete
resource types (e.g., `limiting_resource_adaptor*`, `pool_memory_resource*`) do
**not** implicitly convert to `device_async_resource_ref`.

If you own the function returning a resource pointer, change it to return the
resource by value. This matches RMM's value-resource model and preserves the
resource lifetime through internal shared ownership:

```cpp
// 26.04
rmm::mr::device_memory_resource* get_some_resource();

// 26.06
cuda::mr::any_resource<cuda::mr::device_accessible> get_some_resource();
```

If you cannot change the returning API, dereference the pointer when passing it
to APIs that accept `device_async_resource_ref`:

```cpp
rmm::mr::limiting_resource_adaptor* mr = get_some_resource();
downstream_function(stream, *mr);
```

APIs that take `cuda::mr::any_resource<cuda::mr::device_accessible>` can receive
concrete resource objects directly.

(rmm-2604-2606-resource-cast)=
### Use `resource_cast` Instead of `dynamic_cast`

In 26.04, all resources inherited from `device_memory_resource`, so
`dynamic_cast` could be used to test the runtime type of a resource pointer:

```cpp
// 26.04
auto* mr = rmm::mr::get_current_device_resource();
auto* pool = dynamic_cast<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>*>(mr);
if (pool != nullptr) { /* it's a pool */ }
```

In 26.06, there is no common base class and `get_current_device_resource()` is
removed, so `dynamic_cast` cannot be used. Use `cuda::mr::resource_cast` on a
type-erased resource wrapper such as `device_async_resource_ref` or
`cuda::mr::any_resource<cuda::mr::device_accessible>`:

```cpp
// 26.06
auto ref = rmm::mr::get_current_device_resource_ref();
auto* limiting = cuda::mr::resource_cast<rmm::mr::limiting_resource_adaptor>(&ref);
if (limiting != nullptr) { /* it's a limiting adaptor */ }
```

`resource_cast` uses exact type matching. It returns `nullptr` when the stored
resource has a different concrete type, including a derived or wrapped type.

(rmm-2604-2606-stream-ref-default)=
### `cuda::stream_ref{}` Default Constructor Is Deprecated

The default constructor `cuda::stream_ref{}` is deprecated in CCCL. Use
`cuda::stream_ref{cudaStream_t{nullptr}}` when you need a null/default
stream reference (e.g., in `allocate_sync`/`deallocate_sync` implementations
that delegate to the async methods):

```cpp
// Deprecated — generates a warning
void* allocate_sync(std::size_t bytes, std::size_t alignment) {
  return allocate(cuda::stream_ref{}, bytes, alignment);  // WARNING
}

// Fix
void* allocate_sync(std::size_t bytes, std::size_t alignment) {
  return allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);  // OK
}
```

### Cython

The Python memory resource API is unchanged. This section only applies to
projects that cimport RMM Cython declarations or expose C++ memory resources
through Cython.

(rmm-2604-2606-cython-pxd)=
### Cython `pxd` Declarations Updated

All Cython `pxd` declarations for RMM resources remove the
`(device_memory_resource)` base class and `[Upstream]` template parameter:

```cython
# 26.04
cdef cppclass pool_memory_resource[Upstream](device_memory_resource):
    pool_memory_resource(Upstream* upstream_mr, size_t initial_pool_size,
                         optional[size_t] maximum_pool_size) except +

# 26.06
cdef cppclass pool_memory_resource:
    pool_memory_resource(any_resource[device_accessible] upstream_mr,
                         size_t initial_pool_size,
                         optional[size_t] maximum_pool_size) except +
```

(rmm-2604-2606-cython-downstream-pxd)=
### Downstream Cython `.pxd` Declarations: Pointer to Value Type

Downstream libraries (e.g., cudf) that declare C++ function signatures in
`.pxd` files must change the memory resource parameter from a pointer to the
value type used by the C++ API:

```cython
# 26.04
from rmm.librmm.memory_resource cimport device_memory_resource

cdef extern from "mylib/api.hpp" namespace "mylib" nogil:
    unique_ptr[column] my_function(
        column_view input,
        device_memory_resource *mr
    ) except +

# 26.06
from rmm.librmm.memory_resource cimport device_async_resource_ref

cdef extern from "mylib/api.hpp" namespace "mylib" nogil:
    unique_ptr[column] my_function(
        column_view input,
        device_async_resource_ref mr
    ) except +
```

`device_async_resource_ref` is a lightweight value type (like a pointer),
so it is passed by value rather than by pointer. If the target C++ API takes
an owning `cuda::mr::any_resource<cuda::mr::device_accessible>`, declare the
parameter as `any_resource[device_accessible]` instead and pass `mr.get_mr()`
from `.pyx` code.

(rmm-2604-2606-cython-resource-ref-helpers)=
### Custom Cython Resource Wrappers

If a downstream project wraps its own `shared_resource`-based resource type in
Cython, define an inline C++ helper that returns
`optional[device_async_resource_ref]` from the concrete resource type:

```cython
# my_project/my_resource.pxd
from libcpp.optional cimport optional
from rmm.librmm.memory_resource cimport (
    any_resource,
    device_accessible,
    device_async_resource_ref,
)

cdef extern from "<my_project/my_resource.hpp>" nogil:
    cdef cppclass cpp_MyResource "my_project::MyResource":
        cpp_MyResource(any_resource[device_accessible] upstream) except +
        # ...

cdef extern from *:
    """
    #include <optional>
    #include <rmm/resource_ref.hpp>
    #include <my_project/my_resource.hpp>
    std::optional<rmm::device_async_resource_ref>
    make_my_resource_ref(my_project::MyResource& r) {
        return std::optional<rmm::device_async_resource_ref>(
            rmm::device_async_resource_ref(r));
    }
    """
    optional[device_async_resource_ref] make_my_resource_ref(
        cpp_MyResource&) except +
```

Then in your `.pyx`, after constructing the C++ object, set `c_ref`:

```cython
# my_project/my_resource.pyx
self.c_obj.reset(new cpp_MyResource(upstream_mr.get_mr()))
self.c_ref = make_my_resource_ref(deref(self.c_obj))
```

This pattern matches how RMM's own Python bindings work internally. The
`optional` return type avoids issues with `device_async_resource_ref`'s
non-default-constructible nature during Cython assignment.
