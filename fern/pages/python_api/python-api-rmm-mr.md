---
slug: api-reference/python-api-rmm-mr
---

# rmm.mr

Generated from RMM Python sources.

## `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi`

### `DeviceMemoryResource`

```python
class DeviceMemoryResource
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:12`_

### `allocate`

```python
def allocate(self, nbytes: int, stream: Stream = ...) -> int:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:13`_

### `deallocate`

```python
def deallocate(self, ptr: int, nbytes: int, stream: Stream = ...) -> None:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:14`_

### `UpstreamResourceAdaptor`

```python
class UpstreamResourceAdaptor(DeviceMemoryResource)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:18`_

### `get_upstream`

```python
def get_upstream(self) -> DeviceMemoryResource:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:23`_

### `CudaMemoryResource`

```python
class CudaMemoryResource(DeviceMemoryResource)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:25`_

### `CudaAsyncMemoryResource`

```python
class CudaAsyncMemoryResource(DeviceMemoryResource)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:28`_

### `CudaAsyncViewMemoryResource`

```python
class CudaAsyncViewMemoryResource(DeviceMemoryResource)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:37`_

### `pool_handle`

```python
def pool_handle(self) -> int:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:41`_

### `ManagedMemoryResource`

```python
class ManagedMemoryResource(DeviceMemoryResource)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:43`_

### `SystemMemoryResource`

```python
class SystemMemoryResource(DeviceMemoryResource)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:46`_

### `PinnedHostMemoryResource`

```python
class PinnedHostMemoryResource(DeviceMemoryResource)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:49`_

### `SamHeadroomMemoryResource`

```python
class SamHeadroomMemoryResource(DeviceMemoryResource)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:52`_

### `PoolMemoryResource`

```python
class PoolMemoryResource(UpstreamResourceAdaptor)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:55`_

### `pool_size`

```python
def pool_size(self) -> int:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:62`_

### `ArenaMemoryResource`

```python
class ArenaMemoryResource(UpstreamResourceAdaptor)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:64`_

### `FixedSizeMemoryResource`

```python
class FixedSizeMemoryResource(UpstreamResourceAdaptor)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:72`_

### `BinningMemoryResource`

```python
class BinningMemoryResource(UpstreamResourceAdaptor)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:80`_

### `add_bin`

```python
def add_bin(self, allocation_size: int, bin_resource: DeviceMemoryResource | None = None,) -> None:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:87`_

### `bin_mrs`

```python
def bin_mrs(self) -> list[DeviceMemoryResource]:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:93`_

### `CallbackMemoryResource`

```python
class CallbackMemoryResource(DeviceMemoryResource)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:95`_

### `LimitingResourceAdaptor`

```python
class LimitingResourceAdaptor(UpstreamResourceAdaptor)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:102`_

### `get_allocated_bytes`

```python
def get_allocated_bytes(self) -> int:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:106`_

### `get_allocation_limit`

```python
def get_allocation_limit(self) -> int:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:107`_

### `LoggingResourceAdaptor`

```python
class LoggingResourceAdaptor(UpstreamResourceAdaptor)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:109`_

### `flush`

```python
def flush(self) -> None:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:115`_

### `get_file_name`

```python
def get_file_name(self) -> str:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:116`_

### `StatisticsResourceAdaptor`

```python
class StatisticsResourceAdaptor(UpstreamResourceAdaptor)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:118`_

### `allocation_counts`

```python
def allocation_counts(self) -> Statistics:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:121`_

### `pop_counters`

```python
def pop_counters(self) -> Statistics:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:122`_

### `push_counters`

```python
def push_counters(self) -> Statistics:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:123`_

### `TrackingResourceAdaptor`

```python
class TrackingResourceAdaptor(UpstreamResourceAdaptor)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:125`_

### `get_allocated_bytes`

```python
def get_allocated_bytes(self) -> int:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:129`_

### `get_outstanding_allocations_str`

```python
def get_outstanding_allocations_str(self) -> str:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:130`_

### `log_outstanding_allocations`

```python
def log_outstanding_allocations(self) -> None:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:131`_

### `FailureCallbackResourceAdaptor`

```python
class FailureCallbackResourceAdaptor(UpstreamResourceAdaptor)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:133`_

### `PrefetchResourceAdaptor`

```python
class PrefetchResourceAdaptor(UpstreamResourceAdaptor)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:140`_

### `get_per_device_resource`

```python
def get_per_device_resource(device: int) -> DeviceMemoryResource:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:143`_

### `set_per_device_resource`

```python
def set_per_device_resource(device: int, mr: DeviceMemoryResource) -> None:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:144`_

### `set_current_device_resource`

```python
def set_current_device_resource(mr: DeviceMemoryResource) -> None:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:145`_

### `get_current_device_resource`

```python
def get_current_device_resource() -> DeviceMemoryResource:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:146`_

### `is_initialized`

```python
def is_initialized() -> bool:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:147`_

### `enable_logging`

```python
def enable_logging(log_file_name: str | None = None) -> None:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:148`_

### `disable_logging`

```python
def disable_logging() -> None:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:149`_

### `get_log_filenames`

```python
def get_log_filenames() -> dict[int, str | None]:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:150`_

### `available_device_memory`

```python
def available_device_memory() -> tuple[int, int]:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/_memory_resource.pyi:151`_

## `python/rmm/rmm/pylibrmm/memory_resource/experimental.pyi`

### `CudaAsyncManagedMemoryResource`

```python
class CudaAsyncManagedMemoryResource(DeviceMemoryResource)
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/experimental.pyi:6`_

### `pool_handle`

```python
def pool_handle(self) -> int:
```

_Source: `python/rmm/rmm/pylibrmm/memory_resource/experimental.pyi:8`_
