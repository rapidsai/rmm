# RMM 26.04 to 26.06 Migration Guide

RMM 26.06 completes the migration to CCCL-native memory resources. This release
removes `device_memory_resource`, eliminates the `Upstream` template parameter
from all resources and adaptors, introduces `cuda::mr::shared_resource`-based
ownership, and removes several legacy APIs. This guide covers every breaking
change and shows how to update consuming code.

## Summary of Breaking Changes

| Area | 26.04 | 26.06 |
|------|-------|-------|
| Base class | `device_memory_resource` (virtual) | Removed; resources satisfy `cuda::mr::resource` concept |
| Resource templates | `pool_memory_resource<Upstream>` | `pool_memory_resource` (no template parameter) |
| Upstream parameter | `Upstream*` raw pointer | `device_async_resource_ref` |
| Ownership model | `std::unique_ptr` / raw pointer | `cuda::mr::shared_resource<Impl>` (reference-counted) |
| `allocate` / `deallocate` | `allocate(size, stream)` / `do_allocate` virtual | `allocate(stream, size, alignment)` (CCCL concept) |
| Per-device resource | `set_current_device_resource(device_memory_resource*)` | `set_current_device_resource_ref(device_async_resource_ref)` |
| `owning_wrapper` | Available | Removed |
| `device_memory_resource_view` | Available | Removed |
| Build model | Resources are header-only | Resource implementations compiled into `librmm` |
| Resource ref aliases | Wrappers around CCCL types + legacy bridge | Wrappers around CCCL `resource_ref` / `synchronous_resource_ref` |
| Python `DeviceMemoryResource` | Holds `shared_ptr[device_memory_resource]` | Holds `unique_ptr` to concrete type + `optional[device_async_resource_ref]` |

## C++ Changes

### 1. `device_memory_resource` Base Class Is Removed

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
  [Allocation Signature Changes](#4-allocation-and-deallocation-signature-changes)).
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

  // Declare properties via ADL friend:
  friend void get_property(my_resource const&, cuda::mr::device_accessible) noexcept {}
};
static_assert(cuda::mr::resource_with<my_resource, cuda::mr::device_accessible>);
```

### 2. Template Parameters Removed from All Resources and Adaptors

Every resource and adaptor that previously took an `Upstream` template parameter
is now a non-template class. The upstream is accepted as a
`device_async_resource_ref` instead of a typed pointer.

| 26.04 | 26.06 |
|-------|-------|
| `pool_memory_resource<Upstream>` | `pool_memory_resource` |
| `arena_memory_resource<Upstream>` | `arena_memory_resource` |
| `fixed_size_memory_resource<Upstream>` | `fixed_size_memory_resource` |
| `binning_memory_resource<Upstream>` | `binning_memory_resource` |
| `logging_resource_adaptor<Upstream>` | `logging_resource_adaptor` |
| `tracking_resource_adaptor<Upstream>` | `tracking_resource_adaptor` |
| `statistics_resource_adaptor<Upstream>` | `statistics_resource_adaptor` |
| `aligned_resource_adaptor<Upstream>` | `aligned_resource_adaptor` |
| `limiting_resource_adaptor<Upstream>` | `limiting_resource_adaptor` |
| `thread_safe_resource_adaptor<Upstream>` | `thread_safe_resource_adaptor` |
| `prefetch_resource_adaptor<Upstream>` | `prefetch_resource_adaptor` |
| `callback_memory_resource` | `callback_memory_resource` (unchanged) |
| `failure_callback_resource_adaptor<Upstream>` | `failure_callback_resource_adaptor<ExceptionType>` |

Note: `failure_callback_resource_adaptor` retains a template parameter, but it
is now `ExceptionType` (defaulting to `rmm::out_of_memory`), not `Upstream`.
The upstream is passed as a `device_async_resource_ref`.

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

### 3. Upstream Resources Are Passed by `device_async_resource_ref`, Not Pointer

Constructors for adaptors and pool-based resources now take
`rmm::device_async_resource_ref` (a type-erased non-owning reference) instead
of a raw pointer to the upstream.

A `device_async_resource_ref` is implicitly constructible from any object that
satisfies `cuda::mr::resource_with<T, cuda::mr::device_accessible>`, so you
can pass a resource object directly.

```cpp
// 26.04
rmm::mr::cuda_memory_resource cuda_mr;
rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool{&cuda_mr, pool_size};

// 26.06 — pass the resource directly (implicit conversion to device_async_resource_ref)
rmm::mr::cuda_memory_resource cuda_mr;
rmm::mr::pool_memory_resource pool{cuda_mr, pool_size};
```

### 4. Allocation and Deallocation Signature Changes

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

### 5. Resources Use `cuda::mr::shared_resource` for Ownership

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

### 6. `owning_wrapper` Is Removed

**Header removed:** `rmm/mr/owning_wrapper.hpp`

`owning_wrapper` was used to bundle a resource with ownership of its upstream.
With the `shared_resource` model, resources already share ownership of their
state, so `owning_wrapper` is no longer needed.

```cpp
// 26.04
auto cuda_mr = std::make_shared<rmm::mr::cuda_memory_resource>();
auto pool = rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
    cuda_mr, pool_size);

// 26.06 — resources are value types with shared ownership
rmm::mr::cuda_memory_resource cuda_mr;
rmm::mr::pool_memory_resource pool{cuda_mr, pool_size};
// pool internally shares ownership; no wrapper needed
```

### 7. Per-Device Resource API Changes

The pointer-based per-device resource functions are removed. Use the `_ref`
variants exclusively:

| Removed (26.04) | Replacement (26.06) |
|-----------------|---------------------|
| `get_current_device_resource()` → `device_memory_resource*` | `get_current_device_resource_ref()` → `device_async_resource_ref` |
| `set_current_device_resource(device_memory_resource*)` | `set_current_device_resource_ref(device_async_resource_ref)` |
| `get_per_device_resource(id)` → `device_memory_resource*` | `get_per_device_resource_ref(id)` → `device_async_resource_ref` |
| `set_per_device_resource(id, device_memory_resource*)` | `set_per_device_resource_ref(id, device_async_resource_ref)` |
| — | `reset_per_device_resource_ref(id)` (new) |
| — | `reset_current_device_resource_ref()` (new) |

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
auto prev = rmm::mr::set_current_device_resource_ref(pool);
auto ref = rmm::mr::get_current_device_resource_ref();
// When done, restore: rmm::mr::set_current_device_resource_ref(prev);
// Or reset to default: rmm::mr::reset_current_device_resource_ref();
```

### 8. `device_buffer` Internal Changes

`rmm::device_buffer` now stores a `cuda::mr::any_resource<cuda::mr::device_accessible>`
internally instead of a raw `device_memory_resource*`. The public API is largely
the same, but:

- Constructors accept `device_async_resource_ref` (as before).
- `memory_resource()` returns `rmm::device_async_resource_ref` (was
  `device_memory_resource*` in older releases).

### 9. `device_memory_resource_view` Is Removed

**Header removed:** `rmm/mr/detail/device_memory_resource_view.hpp`

This bridge class was used to wrap a `device_async_resource_ref` into a
`device_memory_resource`. It is no longer needed since the legacy base class
is gone.

### 10. `resource_ref` Type Aliases Updated

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

### 11. `failure_callback_t` Moved to Separate Header

The `failure_callback_t` type alias is now in its own header:

```cpp
// 26.04
#include <rmm/mr/failure_callback_resource_adaptor.hpp>  // included failure_callback_t

// 26.06
#include <rmm/mr/failure_callback_t.hpp>  // for failure_callback_t alone
#include <rmm/mr/failure_callback_resource_adaptor.hpp>  // also includes it
```

### 12. Resource Implementations Are Compiled, Not Header-Only

In 26.04, RMM resource implementations were entirely header-only (template
definitions lived in the headers). In 26.06, the de-templated resources have
their implementations in compiled translation units under `cpp/src/mr/`. The
public headers contain only the class declaration and inline accessors.

This means downstream projects that use RMM memory resources **must link against
`librmm`**. If your project previously relied on RMM being header-only for the
memory resource layer, you will need to update your build system to link the
`rmm` library target.

For CMake users, this should already work if you use `find_package(rmm)` or
`rapids_find_package(rmm)` — the imported `rmm::rmm` target handles linking.

### 13. `rmm::cuda_stream` to `cuda::stream_ref` Conversion

`rmm::cuda_stream::operator cuda::stream_ref()` is no longer `noexcept`. This
is unlikely to require code changes but may affect `noexcept` specifications in
downstream code that converts `cuda_stream` to `stream_ref`.

### 14. Custom Resource Implementation Guide

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
  explicit my_resource(rmm::device_async_resource_ref upstream) : upstream_(upstream) {}

  // Async (stream-ordered) interface
  void* allocate(cuda::stream_ref stream, std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
    return upstream_.allocate(stream, bytes);
  }
  void deallocate(cuda::stream_ref stream, void* p, std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
    upstream_.deallocate(stream, p, bytes);
  }

  // Synchronous interface — required by the CCCL resource concept
  void* allocate_sync(std::size_t bytes,
                      std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
    return allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }
  void deallocate_sync(void* p, std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, p, bytes, alignment);
  }

  bool operator==(my_resource const& other) const noexcept {
    return this == &other;
  }

  friend void get_property(my_resource const&, cuda::mr::device_accessible) noexcept {}

 private:
  rmm::device_async_resource_ref upstream_;
};
static_assert(cuda::mr::resource_with<my_resource, cuda::mr::device_accessible>);
```

> **Note:** The CCCL `resource` concept requires both the async interface
> (`allocate`/`deallocate` with `stream_ref`) and the synchronous interface
> (`allocate_sync`/`deallocate_sync` without a stream). If you omit the
> synchronous methods, `static_assert` will fail. A simple implementation
> delegates to the async methods with a null stream as shown above.

### 14a. Custom Resources Must Be Copyable for `any_resource`

`cuda::mr::any_resource<cuda::mr::device_accessible>` (used internally by
`device_buffer`, per-device resource storage, and as a common owning
type-erased resource) requires that the stored type is both **copyable and
movable**. If your custom resource holds non-copyable members like
`std::shared_mutex`, `std::unique_ptr`, or `std::unordered_set`, it cannot
be stored directly in `any_resource`.

The fix is to wrap non-copyable state in `std::shared_ptr`:

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

### 14b. `get_property` Friend Cannot Be Defined in Local Classes

C++ does not allow `friend` function definitions inside function-scoped
(local) classes. If you define a wrapper struct inside a function body
(common in tests), the `get_property` friend required by the CCCL resource
concept will fail to compile. Move the struct to namespace scope:

```cpp
// Won't compile — friend definition in local class
void my_test() {
  struct local_wrapper {
    void* allocate(...) { ... }
    void deallocate(...) noexcept { ... }
    bool operator==(local_wrapper const&) const noexcept { return true; }
    friend void get_property(local_wrapper const&,
                             cuda::mr::device_accessible) noexcept {}  // ERROR
  };
}

// Fix: move to namespace scope (e.g., anonymous namespace in the test file)
namespace {
struct local_wrapper {
  void* allocate(...) { ... }
  void deallocate(...) noexcept { ... }
  bool operator==(local_wrapper const&) const noexcept { return true; }
  friend void get_property(local_wrapper const&,
                           cuda::mr::device_accessible) noexcept {}  // OK
};
}  // namespace

void my_test() {
  local_wrapper w;
  // ...
}
```

### 14c. Resource Pointers No Longer Convert to `device_async_resource_ref`

In 26.04, `device_async_resource_ref` was implicitly constructible from any
`device_memory_resource*`, so code could pass a raw pointer wherever a ref was
expected. In 26.06, `device_memory_resource` is gone, so pointers to concrete
resource types (e.g., `limiting_resource_adaptor*`, `pool_memory_resource*`) do
**not** implicitly convert to `device_async_resource_ref`.

If you have a function that returns a resource pointer, **dereference** the
pointer when passing it to APIs that accept `device_async_resource_ref`:

```cpp
// 26.04 — implicit conversion from pointer
rmm::mr::limiting_resource_adaptor<rmm::mr::device_memory_resource>* mr = get_some_resource();
rmm::device_buffer buf(size, stream, mr);   // OK: device_memory_resource* -> resource_ref

// 26.06 — dereference the pointer
rmm::mr::limiting_resource_adaptor* mr = get_some_resource();
rmm::device_buffer buf(size, stream, *mr);  // OK: limiting_resource_adaptor& -> resource_ref
//                                    ^ dereference
```

This applies to all APIs accepting `device_async_resource_ref`, including
`device_buffer`, `device_uvector`, `make_device_mdarray`, and any downstream
functions with `device_async_resource_ref` parameters.

> **Tip:** If you see confusing compilation errors about `cuda_stream_view`
> not converting to `std::size_t`, it likely means the compiler skipped a
> `(size, stream, mr)` overload (because `mr` didn't match
> `device_async_resource_ref`) and tried a `(size, alignment, stream, mr)`
> overload instead. The fix is to dereference the pointer or use a ref-returning
> API.

### 14d. `dynamic_cast` on Resources Is No Longer Possible

In 26.04, all resources inherited from `device_memory_resource`, so
`dynamic_cast` could be used to test the runtime type of a resource pointer:

```cpp
// 26.04
auto* mr = rmm::mr::get_current_device_resource();
auto* pool = dynamic_cast<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>*>(mr);
if (pool != nullptr) { /* it's a pool */ }
```

In 26.06, there is no common base class and `get_current_device_resource()` is
removed, so this pattern is impossible. If your code used `dynamic_cast` to
detect the type of the current device resource (e.g., to decide whether to
create a pool), you must find an alternative:

- **Remove the check** — just create the pool unconditionally if your
  application logic requires one.
- **Track resource type explicitly** — use a flag or enum alongside the
  resource to record what was set.
- **Use `any_resource`** — store the resource as
  `cuda::mr::any_resource<cuda::mr::device_accessible>` and track type
  information separately.

### 14e. `const` Data Members Prevent `resource_ref` Construction

The CCCL `resource_ref` (and `any_resource`) requires stored types to satisfy
`semiregular`, which includes copy and move *assignability*. A `const` data
member prevents the compiler from generating copy/move assignment operators,
so a resource with `const` members will fail concept checks even if it is
otherwise well-formed:

```cpp
// Won't work — const member prevents assignment
struct my_resource {
  const std::size_t limit;  // ERROR: makes type non-assignable
  void* allocate(cuda::stream_ref, std::size_t, std::size_t) { ... }
  // ...
};
// static_assert(cuda::mr::resource<my_resource>);  // FAILS

// Fix: remove const
struct my_resource {
  std::size_t limit;  // OK: type is now assignable
  // ...
};
```

This is easy to miss because `const` members don't prevent *construction*
or *copy construction* — only assignment. The error messages from the concept
check can be cryptic (e.g., "type must be movable").

### 14f. Brace Initialization Fails for `shared_resource`-Derived Types

Types inheriting from `cuda::mr::shared_resource<Impl>` cannot be
brace-initialized (`MyResource{args...}`). The `shared_resource` base class
has constructors that prevent aggregate initialization, so brace-init is
ambiguous or fails. Use parenthesized initialization instead:

```cpp
// Won't compile
rmm::mr::pool_memory_resource pool{cuda_mr, initial_size};  // ERROR

// Fix: use parentheses
rmm::mr::pool_memory_resource pool(cuda_mr, initial_size);  // OK
```

This applies to all RMM resource types that use `shared_resource` internally
(which is most of them in 26.06), and to any downstream custom resources
built on `shared_resource`.

### 14g. `cuda::stream_ref{}` Default Constructor Is Deprecated

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

### 14h. `any_resource` Is Fully Type-Erased (No `get<T>()`)

`cuda::mr::any_resource<cuda::mr::device_accessible>` does not provide a
`get<T>()` method to recover the concrete resource type. Once a resource is
stored in `any_resource`, the concrete type is lost.

If you need access to concrete type information after construction (e.g.,
calling `pool_memory_resource::pool_size()` or capturing a
`cuda_async_memory_resource::pool_handle()`), you must extract that
information *before* storing the resource into `any_resource`:

```cpp
// Won't work — can't get concrete type back
cuda::mr::any_resource<cuda::mr::device_accessible> mr =
    rmm::mr::pool_memory_resource(cuda_mr, pool_size);
// mr.get<rmm::mr::pool_memory_resource>()  // NO SUCH METHOD

// Fix: capture what you need before type-erasing
rmm::mr::pool_memory_resource pool(cuda_mr, pool_size);
auto pool_sz = pool.pool_size();  // capture before storing
cuda::mr::any_resource<cuda::mr::device_accessible> mr = std::move(pool);
```

## Python / Cython Changes

### 15. `DeviceMemoryResource` Internal Representation

The base `DeviceMemoryResource` class in Python no longer holds a
`shared_ptr[device_memory_resource]`. Instead, each concrete subclass holds a
`unique_ptr` to its specific C++ type, and the base class holds an
`optional[device_async_resource_ref]` (`c_ref`).

**Cython bindings authors:**
- Replace `self.c_obj.get()` (which returned `device_memory_resource*`) with
  `self.c_ref.value()` (which returns `device_async_resource_ref`).
- The `get_mr()` method is removed.
- Constructing a resource now requires two steps: create the `unique_ptr`
  object, then create the `device_async_resource_ref` from it:

```cython
# 26.04
self.c_obj.reset(new pool_memory_resource[device_memory_resource](
    upstream_mr.get_mr(), initial_pool_size, maximum_pool_size))

# 26.06
self.c_obj.reset(new pool_memory_resource(
    upstream_mr.c_ref.value(), initial_pool_size, maximum_pool_size))
self.c_ref = make_device_async_resource_ref(deref(self.c_obj))
```

### 16. Cython `pxd` Declarations Updated

All Cython `pxd` declarations for RMM resources remove the
`(device_memory_resource)` base class and `[Upstream]` template parameter:

```cython
# 26.04
cdef cppclass pool_memory_resource[Upstream](device_memory_resource):
    pool_memory_resource(Upstream* upstream_mr, size_t initial_pool_size,
                         optional[size_t] maximum_pool_size) except +

# 26.06
cdef cppclass pool_memory_resource:
    pool_memory_resource(device_async_resource_ref upstream_mr,
                         size_t initial_pool_size,
                         optional[size_t] maximum_pool_size) except +
```

### 17. Per-Device Resource Python API

The Python `set_per_device_resource` and `set_current_device_resource`
functions now call `set_per_device_resource_ref` / `set_current_device_resource_ref`
under the hood, passing `device_async_resource_ref` instead of
`device_memory_resource*`. The Python-level API is unchanged for users.

### 18. `failure_callback_resource_adaptor` Cython Template Change

The Cython template parameter changes from `Upstream` to `ExceptionType`:

```cython
# 26.04
cdef cppclass failure_callback_resource_adaptor[Upstream](device_memory_resource):
    failure_callback_resource_adaptor(Upstream* upstream_mr, ...) except +

# 26.06
cdef cppclass failure_callback_resource_adaptor[ExceptionType]:
    failure_callback_resource_adaptor(device_async_resource_ref upstream_mr, ...) except +

ctypedef failure_callback_resource_adaptor[out_of_memory] failure_callback_resource_adaptor_oom
```

### 19. Downstream Cython `.pxd` Declarations: Pointer to Value Type

Downstream libraries (e.g., cudf) that declare C++ function signatures in
`.pxd` files must change the memory resource parameter from a pointer to a
value type:

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
so it is passed by value rather than by pointer.

### 20. Downstream Cython `.pyx` Files: `mr.get_mr()` to `mr.c_ref.value()`

In `.pyx` files that call C++ functions, replace `mr.get_mr()` with
`mr.c_ref.value()`. The `c_ref` attribute is a `cdef` (C-level) attribute
on `DeviceMemoryResource`, so accessing `mr.c_ref.value()` does **not**
require the GIL and can be used directly inside `with nogil:` blocks:

```cython
# 26.04
cdef DeviceMemoryResource mr = ...
with nogil:
    result = cpp_my_function(input.view(), mr.get_mr())

# 26.06
cdef DeviceMemoryResource mr = ...
with nogil:
    result = cpp_my_function(input.view(), mr.c_ref.value())
```

This is a mechanical replacement. Search for `\.get_mr()` across all `.pyx`
files and replace with `.c_ref.value()`.

### 21. Downstream Cython Wrappers Need `make_*_resource_ref` Helpers

RMM's Python bindings include inline C++ helper functions
(`make_device_async_resource_ref`) that produce
`optional[device_async_resource_ref]` from each concrete RMM resource type.
These are needed because Cython cannot invoke C++ implicit conversion
operators directly.

Downstream projects that wrap their own `shared_resource`-based types in
Cython must define similar helpers. Define an inline C++ function in a
`cdef extern from *` block in your `.pxd` file:

```cython
# my_project/my_resource.pxd
from libcpp.optional cimport optional
from rmm.librmm.memory_resource cimport device_async_resource_ref

cdef extern from "<my_project/my_resource.hpp>" nogil:
    cdef cppclass cpp_MyResource "my_project::MyResource":
        cpp_MyResource(device_async_resource_ref upstream) except +
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
self.c_obj.reset(new cpp_MyResource(upstream_mr.c_ref.value()))
self.c_ref = make_my_resource_ref(deref(self.c_obj))
```

This pattern matches how RMM's own Python bindings work internally. The
`optional` return type avoids issues with `device_async_resource_ref`'s
non-default-constructible nature during Cython assignment.

## Removed Headers

| Removed Header | Replacement |
|----------------|-------------|
| `rmm/mr/device_memory_resource.hpp` | Implement `cuda::mr::resource` concept directly |
| `rmm/mr/owning_wrapper.hpp` | Resources are value types with shared ownership |
| `rmm/mr/detail/device_memory_resource_view.hpp` | No replacement needed |

## Migration Checklist

1. **Remove `device_memory_resource` includes and inheritance** from custom resources.
2. **Remove `Upstream` template parameters** from all RMM resource type names.
3. **Pass upstream resources by value** (or as `device_async_resource_ref`) instead of pointer.
4. **Update `allocate`/`deallocate` calls** to the new argument order: `(stream, bytes[, alignment])`.
5. **Replace `do_allocate`/`do_deallocate` overrides** with public `allocate`/`deallocate` methods
   matching the CCCL resource concept.
6. **Add `allocate_sync`/`deallocate_sync`** to custom resources (required by CCCL concept).
7. **Replace pointer-based per-device resource functions** with `_ref` variants.
8. **Remove `owning_wrapper` usage**; resources are copyable and share ownership.
9. **Ensure custom resources are copyable** if stored in `any_resource`; wrap non-copyable
   members (mutexes, unique_ptrs) in `std::shared_ptr`.
10. **Move resource structs with `get_property` friends out of function scope** to namespace scope.
11. **Dereference resource pointers** when passing to APIs expecting `device_async_resource_ref`.
12. **Remove `dynamic_cast` on resource pointers**; there is no common base class.
13. **Update Cython `.pxd` files**: `device_memory_resource *mr` → `device_async_resource_ref mr`.
14. **Update Cython `.pyx` files**: `mr.get_mr()` → `mr.c_ref.value()`.
15. **Update `#include` directives** for removed headers.
16. **Link against `librmm`** if your project previously used RMM resources
    as header-only; resource implementations are now compiled.
17. **Verify** with `static_assert(cuda::mr::resource_with<MyResource, cuda::mr::device_accessible>)`
    that custom resources satisfy the CCCL concept.
18. **Remove `const` from data members** in custom resources if the type must be assignable
    (required by `resource_ref` and `any_resource`).
19. **Use parenthesized initialization** for `shared_resource`-derived types (`Resource(args...)`,
    not `Resource{args...}`).
20. **Replace `cuda::stream_ref{}`** with `cuda::stream_ref{cudaStream_t{nullptr}}` to avoid
    deprecation warnings.
21. **Capture concrete type info before type-erasing** into `any_resource` — there is no
    `get<T>()` to recover the concrete type.
22. **Add `make_*_resource_ref` inline C++ helpers** in downstream Cython `.pxd` files for
    custom resource types that need `c_ref` assignment.
