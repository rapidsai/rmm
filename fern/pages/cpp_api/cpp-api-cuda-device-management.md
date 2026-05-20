---
slug: api-reference/cpp-api-cuda-device-management
---

# CUDA Device Management

Generated from RMM C++ headers.

## `cpp/include/rmm/cuda_device.hpp`

### CUDA Device Id Struct

Strong type for a CUDA device identifier.

```cpp
struct cuda_device_id
```

_Source: `cpp/include/rmm/cuda_device.hpp:27`_

### CUDA Device Id Constructor (cuda_device.hpp:33)

Construct a `cuda_device_id` from the current device

```cpp
cuda_device_id() noexcept : id_
```

_Source: `cpp/include/rmm/cuda_device.hpp:33`_

### CUDA Device Id Constructor (cuda_device.hpp:40)

Construct a `cuda_device_id` from the specified integer value.

**Parameters:**

- `dev_id`: The device's integer identifier

```cpp
explicit constexpr cuda_device_id(value_type dev_id) noexcept : id_
```

_Source: `cpp/include/rmm/cuda_device.hpp:40`_

### Get Current CUDA Device

Returns a `cuda_device_id` for the current device

The current device is the device on which the calling thread executes device code.

**Returns:** `cuda_device_id` for the current device

```cpp
cuda_device_id get_current_cuda_device();
```

_Source: `cpp/include/rmm/cuda_device.hpp:85`_

### Get Num CUDA Devices

Returns the number of CUDA devices in the system

**Returns:** Number of CUDA devices in the system

```cpp
int get_num_cuda_devices();
```

_Source: `cpp/include/rmm/cuda_device.hpp:92`_

### Percent Of Free Device Memory

Returns the approximate specified percent of available device memory on the current CUDA device, aligned (down) to the nearest CUDA allocation size.

**Parameters:**

- `percent`: The percent of free memory to return.

**Returns:** The recommended initial device memory pool size in bytes.

```cpp
std::size_t percent_of_free_device_memory(int percent);
```

_Source: `cpp/include/rmm/cuda_device.hpp:109`_

### CUDA Set Device Raii Struct

RAII class that sets the current CUDA device to the specified device on construction and restores the previous device on destruction.

```cpp
struct cuda_set_device_raii
```

_Source: `cpp/include/rmm/cuda_device.hpp:115`_

### CUDA Set Device Raii Constructor

Construct a new cuda_set_device_raii object and sets the current CUDA device to `dev_id`

**Parameters:**

- `dev_id`: The device to set as the current CUDA device

```cpp
explicit cuda_set_device_raii(cuda_device_id dev_id);
```

_Source: `cpp/include/rmm/cuda_device.hpp:121`_

## `cpp/include/rmm/prefetch.hpp`

### Prefetch (prefetch.hpp:37)

Prefetch memory to the specified device on the specified stream.

This function is a no-op if the pointer is not to CUDA managed memory or if concurrent managed access is not supported.

**Throws:**

- `rmm::cuda_error`: if the prefetch fails.

**Parameters:**

- `ptr`: The pointer to the memory to prefetch
- `size`: The number of bytes to prefetch
- `device`: The device to prefetch to
- `stream`: The stream to use for the prefetch

```cpp
void prefetch(void const* ptr, std::size_t size, rmm::cuda_device_id device, rmm::cuda_stream_view stream);
```

_Source: `cpp/include/rmm/prefetch.hpp:37`_

### Prefetch (prefetch.hpp:53)

Prefetch a span of memory to the specified device on the specified stream.

This function is a no-op if the buffer is not backed by CUDA managed memory.

**Throws:**

- `rmm::cuda_error`: if the prefetch fails.

**Parameters:**

- `data`: The span to prefetch
- `device`: The device to prefetch to
- `stream`: The stream to use for the prefetch

```cpp
template <typename T> void prefetch(cuda::std::span<T const> data, rmm::cuda_device_id device, rmm::cuda_stream_view stream)
```

_Source: `cpp/include/rmm/prefetch.hpp:53`_

## `cpp/include/rmm/process_is_exiting.hpp`

### Process Is Exiting

Returns `true` if the process has entered `exit()` / atexit handler execution.

Destructors of static objects, as well as atexit handlers registered by other DSOs, run during process termination after `main()` has returned. At that point calling into the CUDA runtime or driver is undefined behavior: the primary context may already be destroyed, and CUDA API calls may dereference released state and crash inside libcuda rather than returning an error.

Use this function from a memory resource destructor (or a helper invoked by a destructor, such as a `release()` method) when the resource may be held in RMM's internal per-device resource map and destroyed during process termination. In that case the destructor may run after the CUDA primary context has been destroyed, and calling into the CUDA runtime is undefined behavior. Destructors can avoid that by:

1. Never calling CUDA APIs from the destructor at all, or 2. Consulting `rmm::process_is_exiting()` in the destructor (and in any helper invoked by the destructor, such as a `release()` method) and skipping CUDA API calls when it returns `true`. In that case, resources that would have been explicitly released should be leaked; the OS reclaims them when the process exits.

Storing RMM objects with static or thread-local scope is unsupported. Users should not create their own static containers of RMM objects and rely on `rmm::process_is_exiting()` to make those destructors safe.

Calling `rmm::process_is_exiting()` from a resource destructor is always safe: it performs a single atomic load (acquire semantics) and never calls into CUDA.

Example:

```cpp
class my_resource final : public ... {
  ~my_resource() override
  {
    if (rmm::process_is_exiting()) {
      return;
    }
    RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaFree(ptr_));
  }
};
```

**Returns:** `true` if `exit()` has begun; `false` otherwise.

```cpp
bool process_is_exiting() noexcept;
```

_Source: `cpp/include/rmm/process_is_exiting.hpp:60`_
