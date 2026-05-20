---
slug: api-reference/cpp-api-thrust-integrations
---

# Thrust Integrations

Generated from RMM C++ headers.

## `cpp/include/rmm/exec_policy.hpp`

### Thrust Exec Policy T Type Alias

Synchronous execution policy for allocations using Thrust

```cpp
using thrust_exec_policy_t = thrust::detail::execute_with_allocator<mr::thrust_allocator<char>, thrust::cuda_cub::execute_on_stream_base>;
```

_Source: `cpp/include/rmm/exec_policy.hpp:32`_

### Exec Policy Class

Helper class usable as a Thrust CUDA execution policy that uses RMM for temporary memory allocation on the specified stream.

```cpp
class exec_policy : public thrust_exec_policy_t
```

_Source: `cpp/include/rmm/exec_policy.hpp:40`_

### Exec Policy Constructor

Construct a new execution policy object

**Parameters:**

- `stream`: The stream on which to allocate temporary memory
- `mr`: The resource to use for allocating temporary memory

```cpp
explicit exec_policy( cuda_stream_view stream = cuda_stream_default, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref());
```

_Source: `cpp/include/rmm/exec_policy.hpp:48`_

### Thrust Exec Policy Nosync T Type Alias

Asynchronous execution policy for allocations using Thrust

```cpp
using thrust_exec_policy_nosync_t = thrust::detail::execute_with_allocator<mr::thrust_allocator<char>, thrust::cuda_cub::execute_on_stream_nosync_base>;
```

_Source: `cpp/include/rmm/exec_policy.hpp:56`_

### Exec Policy Nosync Class

Helper class usable as a Thrust CUDA execution policy that uses RMM for temporary memory allocation on the specified stream and which allows the Thrust backend to skip stream synchronizations that are not required for correctness.

```cpp
class exec_policy_nosync : public thrust_exec_policy_nosync_t
```

_Source: `cpp/include/rmm/exec_policy.hpp:66`_

### Exec Policy Nosync Constructor

Construct a new execution policy object

**Parameters:**

- `stream`: The stream on which to allocate temporary memory
- `mr`: The resource to use for allocating temporary memory

```cpp
explicit exec_policy_nosync( cuda_stream_view stream = cuda_stream_default, cuda::mr::any_resource<cuda::mr::device_accessible> mr = mr::get_current_device_resource_ref());
```

_Source: `cpp/include/rmm/exec_policy.hpp:74`_
