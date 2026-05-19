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

```cpp
rapids_logger::sink_ptr default_sink();
```

_Source: `cpp/include/rmm/logger.hpp:23`_

### Default Pattern

Returns the default log pattern for the global logger.

```cpp
std::string default_pattern();
```

_Source: `cpp/include/rmm/logger.hpp:30`_

### Default Logger

Get the default logger.

```cpp
rapids_logger::logger& default_logger();
```

_Source: `cpp/include/rmm/logger.hpp:37`_

## `cpp/include/rmm/detail/runtime_capabilities.hpp`

### Runtime Async Alloc

Determine at runtime if the CUDA driver supports the stream-ordered

```cpp
struct runtime_async_alloc {
```

_Source: `cpp/include/rmm/detail/runtime_capabilities.hpp:36`_

### Export Handle Type

Check whether the specified `cudaMemAllocationHandleType` is supported on the present

```cpp
struct export_handle_type {
```

_Source: `cpp/include/rmm/detail/runtime_capabilities.hpp:58`_

### Concurrent Managed Access

Check if the current device supports concurrent managed access.

```cpp
struct concurrent_managed_access {
```

_Source: `cpp/include/rmm/detail/runtime_capabilities.hpp:119`_

### Runtime Async Managed Alloc

Determine at runtime if the CUDA driver/runtime supports the stream-ordered

```cpp
struct runtime_async_managed_alloc {
```

_Source: `cpp/include/rmm/detail/runtime_capabilities.hpp:139`_

### Device Integrated Memory

Check if the current device is an integrated memory system.

```cpp
struct device_integrated_memory {
```

_Source: `cpp/include/rmm/detail/runtime_capabilities.hpp:163`_
