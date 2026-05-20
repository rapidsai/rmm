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

```cpp
explicit constexpr cuda_device_id(value_type dev_id) noexcept : id_
```

_Source: `cpp/include/rmm/cuda_device.hpp:40`_

### Get Current CUDA Device

Returns a `cuda_device_id` for the current device

```cpp
cuda_device_id get_current_cuda_device();
```

_Source: `cpp/include/rmm/cuda_device.hpp:85`_

### Get Num CUDA Devices

Returns the number of CUDA devices in the system

```cpp
int get_num_cuda_devices();
```

_Source: `cpp/include/rmm/cuda_device.hpp:92`_

### Percent Of Free Device Memory

Returns the approximate specified percent of available device memory on the current CUDA device, aligned (down) to the nearest CUDA allocation size.

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

```cpp
explicit cuda_set_device_raii(cuda_device_id dev_id);
```

_Source: `cpp/include/rmm/cuda_device.hpp:121`_

## `cpp/include/rmm/prefetch.hpp`

### Prefetch (prefetch.hpp:37)

Prefetch memory to the specified device on the specified stream.

```cpp
void prefetch(void const* ptr, std::size_t size, rmm::cuda_device_id device, rmm::cuda_stream_view stream);
```

_Source: `cpp/include/rmm/prefetch.hpp:37`_

### Prefetch (prefetch.hpp:53)

Prefetch a span of memory to the specified device on the specified stream.

```cpp
template <typename T> void prefetch(cuda::std::span<T const> data, rmm::cuda_device_id device, rmm::cuda_stream_view stream)
```

_Source: `cpp/include/rmm/prefetch.hpp:53`_

## `cpp/include/rmm/process_is_exiting.hpp`

### Process Is Exiting

Returns `true` if the process has entered `exit()` / atexit handler execution.

```cpp
bool process_is_exiting() noexcept;
```

_Source: `cpp/include/rmm/process_is_exiting.hpp:60`_
