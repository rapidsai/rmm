# Build Error Analysis: Recursive Constraint Satisfaction in CCCL 3.2

## The Error

When building RMM with CCCL 3.2, the compiler fails with:

```
error: satisfaction of atomic constraint '...' depends on itself
```

This occurs in `resource_ref_conversion_tests.cpp:23` when attempting:
```cpp
rmm::device_async_resource_ref d_ref{hd_ref};
```

Where `hd_ref` is `rmm::host_device_async_resource_ref`.

---

## Root Cause: Recursive Concept Constraint Evaluation

### Background: CCCL 3.2's `basic_any` Architecture

CCCL 3.2 uses a sophisticated type-erased polymorphism system based on `__basic_any`. The `resource_ref` types (`cuda::mr::resource_ref`) are built on top of this:

- `cuda::mr::resource_ref<device_accessible>` → type-erased ref that can hold any resource satisfying device_accessible
- `cuda::mr::resource_ref<host_accessible, device_accessible>` → same, but requires both properties

When constructing a `basic_any` from a concrete type, CCCL must verify the type "satisfies" the required interfaces. This happens via the `__satisfies` concept:

```cpp
template<class _Tp, class _Interface, class UnsatisfiedInterface>
concept __satisfies = __has_overrides<_Tp, UnsatisfiedInterface>;
```

### The Recursive Loop

The problem occurs because `cccl_async_resource_ref` in `cpp/include/rmm/detail/cccl_adaptors.hpp` has these templated converting constructors:

```cpp
// Line 123-127 in cccl_resource_ref
template <typename OtherResourceType>
cccl_resource_ref(cccl_resource_ref<OtherResourceType> const& other)
  : view_{other.view_}, ref_{view_.has_value() ? wrapped_type{*view_} : wrapped_type{other.ref_}}

// Line 274-278 in cccl_async_resource_ref
template <typename OtherResourceType>
cccl_async_resource_ref(cccl_async_resource_ref<OtherResourceType> const& other)
  : base(static_cast<cccl_resource_ref<OtherResourceType> const&>(other))
```

Here's what happens when the compiler processes line 23 of the test:

1. **User writes**: `rmm::device_async_resource_ref d_ref{hd_ref};`
   - `d_ref` = `cccl_async_resource_ref<resource_ref<device_accessible>>`
   - `hd_ref` = `cccl_async_resource_ref<resource_ref<host_accessible, device_accessible>>`

2. **Compiler looks for matching constructor** in `cccl_async_resource_ref`

3. **Constructor from base's `ResourceType const&`** is inherited:
   ```cpp
   cccl_async_resource_ref(ResourceType const& ref)  // ResourceType = resource_ref<device_accessible>
   ```

4. **CCCL's `resource_ref` constructor** requires checking if `hd_ref` (which is `cccl_async_resource_ref<...>`) satisfies the `__iset_<device_accessible, ...>` interface

5. **`__satisfies` concept evaluation** begins:
   - Checks if `cccl_async_resource_ref<host_device_ref>` has proper `overrides`
   - This requires checking `__has_overrides<cccl_async_resource_ref<...>, Interface>`

6. **To check `__has_overrides`**, the compiler must examine all constructors of `cccl_async_resource_ref<host_device_ref>` to understand what types it can be constructed from

7. **The templated converting constructor** is examined:
   ```cpp
   template <typename OtherResourceType>
   cccl_async_resource_ref(cccl_async_resource_ref<OtherResourceType> const& other)
   ```

8. **This constructor's constraints** trigger evaluation of whether `cccl_async_resource_ref<SomeOther>` satisfies certain interfaces

9. **RECURSIVE LOOP**: Step 8 triggers the same constraint checking as Step 5, but with different template arguments. The compiler detects this self-dependency and errors out.

### Visual Flow:

```
d_ref{hd_ref}
    ↓
Check: Does hd_ref satisfy device_async_resource_ref's interface?
    ↓
Check: __satisfies<cccl_async_resource_ref<HD_type>, Interface>
    ↓
Check: __has_overrides<cccl_async_resource_ref<HD_type>>
    ↓
Examine constructors of cccl_async_resource_ref<HD_type>
    ↓
Found: template<typename Other> cccl_async_resource_ref(cccl_async_resource_ref<Other> const&)
    ↓
Check: Does cccl_async_resource_ref<Other> satisfy interface?  ← RECURSIVE!
    ↓
ERROR: Constraint satisfaction depends on itself
```

---

## Why This Happens in CCCL 3.2 but Not Earlier

CCCL 3.2 introduced a new `basic_any`-based implementation for `resource_ref`. The previous implementation likely used:
- Simpler type erasure without concept-based interface checking
- Or less aggressive constraint evaluation

The new architecture's power (compile-time interface checking) is also its weakness when wrappers with templated constructors are involved.

---

## The Inheritance Amplifies the Problem

The current design uses inheritance:
```cpp
class cccl_async_resource_ref : public cccl_resource_ref<ResourceType>
```

With `using base::base;` inheriting all base constructors, plus its own templated constructor. This creates multiple paths for the compiler to explore, all leading to the recursive constraint check.

---

## Potential Fix Strategies

### Option A: Add SFINAE/Requires to Break Recursion

Add constraints to the templated constructor that prevent it from being considered during certain concept checks:

```cpp
template <typename OtherResourceType>
  requires(!std::is_same_v<OtherResourceType, ResourceType> &&
           std::is_constructible_v<ResourceType, OtherResourceType const&>)
cccl_async_resource_ref(cccl_async_resource_ref<OtherResourceType> const& other)
```

This might not fully solve the issue since `std::is_constructible_v` could itself trigger recursive checks.

### Option B: Use Explicit Named Conversion Function

Instead of implicit conversion via constructor:
```cpp
// Remove the templated constructor
// Add explicit conversion:
template <typename Target>
Target to() const { return Target{ref_}; }
```

Usage: `rmm::device_async_resource_ref d_ref = hd_ref.to<rmm::device_async_resource_ref>();`

### Option C: Don't Wrap CCCL Types

Use CCCL's `resource_ref` types directly and handle the `device_memory_resource*` conversion separately:
- RMM's `resource_ref` types = direct aliases to CCCL types
- Separate adapter functions for DMR* conversion

### Option D: Use Composition Differently

Store the wrapped `ResourceType` without making the wrapper itself look like a resource_ref to CCCL's type system. This would require not relying on CCCL's implicit conversion mechanisms.

---

## Files Involved

| File | Role |
|------|------|
| `cpp/include/rmm/detail/cccl_adaptors.hpp` | Defines the problematic wrapper classes |
| `cpp/include/rmm/resource_ref.hpp` | Type aliases for RMM resource_ref types |
| `cpp/tests/resource_ref_conversion_tests.cpp` | Test file triggering the error |

---

## Summary

The error is a **fundamental incompatibility** between:
1. RMM's wrapper classes (`cccl_resource_ref`, `cccl_async_resource_ref`) with templated converting constructors
2. CCCL 3.2's `basic_any`-based `resource_ref` that uses recursive concept checking

The wrapper's templated constructors create infinite recursion when CCCL's constraint machinery tries to determine if the wrapper type satisfies its required interfaces. This is a known limitation of C++20 concepts when combined with certain template patterns.
