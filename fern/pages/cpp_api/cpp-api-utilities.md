---
slug: api-reference/cpp-api-utilities
---

# Utilities

Generated from RMM C++ headers.

## `cpp/include/rmm/aligned.hpp`

No documented declarations found.

## `cpp/include/rmm/logger.hpp`

### Default Sink

Returns the default sink for the global logger.

If the environment variable `RMM_DEBUG_LOG_FILE` is defined, the default sink is a sink to that file. Otherwise, the default is to dump to stderr.

**Returns:** sink_ptr The sink to use

```cpp
rapids_logger::sink_ptr default_sink();
```

_Source: `cpp/include/rmm/logger.hpp:23`_

### Default Pattern

Returns the default log pattern for the global logger.

**Returns:** std::string The default log pattern.

```cpp
std::string default_pattern();
```

_Source: `cpp/include/rmm/logger.hpp:30`_

### Default Logger

Get the default logger.

**Returns:** logger& The default logger

```cpp
rapids_logger::logger& default_logger();
```

_Source: `cpp/include/rmm/logger.hpp:37`_

## `cpp/include/rmm/detail/runtime_capabilities.hpp`

### Runtime Async Alloc Struct

Determine at runtime if the CUDA driver supports the stream-ordered memory allocator functions.

Stream-ordered memory pools were introduced in CUDA 11.2. This allows RMM users to compile/link against newer CUDA versions and run with older drivers.

```cpp
struct runtime_async_alloc
```

_Source: `cpp/include/rmm/detail/runtime_capabilities.hpp:36`_

### Export Handle Type Struct

Check whether the specified `cudaMemAllocationHandleType` is supported on the present CUDA driver/runtime version.

**Parameters:**

- `handle_type`: An IPC export handle type to check for support.

**Returns:** true if supported

**Returns:** false if unsupported

```cpp
struct export_handle_type
```

_Source: `cpp/include/rmm/detail/runtime_capabilities.hpp:58`_

### Concurrent Managed Access Struct

Check if the current device supports concurrent managed access. Concurrent managed access is required for prefetching to work.

**Returns:** true if the device supports concurrent managed access, false otherwise

```cpp
struct concurrent_managed_access
```

_Source: `cpp/include/rmm/detail/runtime_capabilities.hpp:119`_

### Runtime Async Managed Alloc Struct

Determine at runtime if the CUDA driver/runtime supports the stream-ordered managed memory allocator functions.

Stream-ordered managed memory pools were introduced in CUDA 13.0.

```cpp
struct runtime_async_managed_alloc
```

_Source: `cpp/include/rmm/detail/runtime_capabilities.hpp:139`_

### Device Integrated Memory Struct

Check if the current device is an integrated memory system.

**Returns:** true if the device is an integrated memory system, false otherwise

```cpp
struct device_integrated_memory
```

_Source: `cpp/include/rmm/detail/runtime_capabilities.hpp:163`_
