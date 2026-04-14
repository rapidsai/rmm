# Programming Guide

This guide covers using RMM in C++ and Python applications, including memory resources, containers, and library integrations.

## Basic Example

`````{tabs}
````{code-tab} c++
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/device_buffer.hpp>
#include <iostream>

int main() {
    // Create a memory resource
    rmm::mr::cuda_async_memory_resource async_mr;

    // Allocate device memory using the resource
    rmm::cuda_stream stream;
    rmm::device_buffer buffer(1024, stream.view(), async_mr);

    std::cout << "Allocated " << buffer.size() << " bytes\n";

    return 0;
}
````
````{code-tab} python
import rmm

# Create a memory resource
mr = rmm.mr.CudaAsyncMemoryResource()

# Allocate device memory using the resource
buffer = rmm.DeviceBuffer(size=1024, mr=mr)

print(f"Allocated {buffer.size} bytes at {hex(buffer.ptr)}")
````
`````

## Memory Resources

Memory resources control how device memory is allocated. RMM provides several resource types optimized for different use cases.

### Explicit Resource Passing

The preferred way to use a memory resource is to pass it explicitly when allocating memory. This makes it clear which resource handles each allocation:

`````{tabs}
````{group-tab} C++
```{literalinclude} ../../cpp/examples/docs/src/guide.cu
---
language: cuda
start-after: "// [explicit-resource]"
end-before: "// [/explicit-resource]"
dedent:
---
```
````
````{group-tab} Python
```{literalinclude} ../../python/rmm/rmm/tests/examples/guide.py
---
language: python
start-after: "# [explicit-resource]"
end-before: "# [/explicit-resource]"
dedent:
---
```
````
`````

### Setting the Current Device Resource

RMM also provides a global "current device resource" that is used when no resource is passed explicitly:

`````{tabs}
````{group-tab} C++
```{literalinclude} ../../cpp/examples/docs/src/guide.cu
---
language: cuda
start-after: "// [current-device-resource]"
end-before: "// [/current-device-resource]"
dedent:
---
```
````
````{group-tab} Python
```{literalinclude} ../../python/rmm/rmm/tests/examples/guide.py
---
language: python
start-after: "# [current-device-resource]"
end-before: "# [/current-device-resource]"
dedent:
---
```
````
`````

> **Warning**: The default resource must be set **before** allocating any device memory on that device. Setting or changing the resource after device allocations have been made can lead to unexpected behavior or crashes.

### Available Resources

RMM provides base memory resources (e.g., {py:class}`~rmm.mr.CudaAsyncMemoryResource`, {py:class}`~rmm.mr.ManagedMemoryResource`) and resource adaptors (e.g., {py:class}`~rmm.mr.PoolMemoryResource`, {py:class}`~rmm.mr.StatisticsResourceAdaptor`) that wrap an upstream resource to add functionality. See [Choosing a Memory Resource](choosing_memory_resources.md) for recommendations and the API references ([C++ memory resources](../cpp/memory_resources/memory_resources.md), [C++ adaptors](../cpp/memory_resources/memory_resource_adaptors.md), [Python](../python/mr.md)) for the full list.

## Containers

RMM provides RAII containers that automatically manage device memory lifetime.

### DeviceBuffer

Untyped, uninitialized device memory ({cpp:class}`C++ <rmm::device_buffer>`, {py:class}`Python <rmm.DeviceBuffer>`):

`````{tabs}
````{group-tab} C++
```{literalinclude} ../../cpp/examples/docs/src/guide.cu
---
language: cuda
start-after: "// [device-buffer]"
end-before: "// [/device-buffer]"
dedent:
---
```
````
````{group-tab} Python
```{literalinclude} ../../python/rmm/rmm/tests/examples/guide.py
---
language: python
start-after: "# [device-buffer]"
end-before: "# [/device-buffer]"
dedent:
---
```
````
`````

### device_uvector (C++)

Typed, uninitialized device vector for trivially copyable types ({cpp:class}`API <rmm::device_uvector>`):

```{literalinclude} ../../cpp/examples/docs/src/guide.cu
---
language: cuda
start-after: "// [device-uvector]"
end-before: "// [/device-uvector]"
dedent:
---
```

### device_scalar (C++)

Single typed element with host-device transfer convenience ({cpp:class}`API <rmm::device_scalar>`):

```{literalinclude} ../../cpp/examples/docs/src/guide.cu
---
language: cuda
start-after: "// [device-scalar]"
end-before: "// [/device-scalar]"
dedent:
---
```

## Resource Adaptors

Adaptors wrap resources to add functionality like statistics tracking and logging.

### Statistics Tracking

`````{tabs}
````{group-tab} C++
```{literalinclude} ../../cpp/examples/docs/src/guide.cu
---
language: cuda
start-after: "// [statistics-tracking]"
end-before: "// [/statistics-tracking]"
dedent:
---
```
````
````{group-tab} Python
```{literalinclude} ../../python/rmm/rmm/tests/examples/guide.py
---
language: python
start-after: "# [statistics-tracking]"
end-before: "# [/statistics-tracking]"
dedent:
---
```
````
`````

### Logging

`````{tabs}
````{group-tab} C++
```{literalinclude} ../../cpp/examples/docs/src/guide.cu
---
language: cuda
start-after: "// [logging]"
end-before: "// [/logging]"
dedent:
---
```
````
````{group-tab} Python
```{literalinclude} ../../python/rmm/rmm/tests/examples/guide.py
---
language: python
start-after: "# [logging]"
end-before: "# [/logging]"
dedent:
---
```
````
`````

CSV format: `Thread,Time,Action,Pointer,Size,Stream`

See [Logging and Profiling](logging.md) for more details.

### Composing Resources

Adaptors can be stacked to combine functionality:

`````{tabs}
````{group-tab} C++
```{literalinclude} ../../cpp/examples/docs/src/guide.cu
---
language: cuda
start-after: "// [composing-resources]"
end-before: "// [/composing-resources]"
dedent:
---
```
````
````{group-tab} Python
```{literalinclude} ../../python/rmm/rmm/tests/examples/guide.py
---
language: python
start-after: "# [composing-resources]"
end-before: "# [/composing-resources]"
dedent:
---
```
````
`````

Order matters: outer adaptors see all allocations from inner resources.

## Library Integrations

### Thrust (C++)

Use {cpp:class}`rmm::exec_policy_nosync` to make Thrust algorithms use RMM for temporary storage. Passing the resource explicitly makes it clear which resource handles temporaries:

```{literalinclude} ../../cpp/examples/docs/src/guide.cu
---
language: cuda
start-after: "// [thrust]"
end-before: "// [/thrust]"
dedent:
---
```

`exec_policy_nosync` allows the Thrust backend to skip stream synchronizations that are not required for correctness, improving performance. Stream-ordered applications using RMM should always prefer `exec_policy_nosync`. If stream synchronizations are required, the application should insert them explicitly before reading device data from the host.

### CuPy (Python)

Configure CuPy to use RMM for all device memory allocations ({py:func}`API <rmm.allocators.cupy.rmm_cupy_allocator>`):

```{literalinclude} ../../python/rmm/rmm/tests/examples/guide.py
---
language: python
start-after: "# [cupy]"
end-before: "# [/cupy]"
dedent:
---
```

### Numba (Python)

Configure Numba to use RMM for device memory in CUDA JIT-compiled functions ({py:class}`API <rmm.allocators.numba.RMMNumbaManager>`):

```{literalinclude} ../../python/rmm/rmm/tests/examples/guide.py
---
language: python
start-after: "# [numba]"
end-before: "# [/numba]"
dedent:
---
```

Or use the environment variable:

```bash
NUMBA_CUDA_MEMORY_MANAGER=rmm.allocators.numba python script.py
```

### PyTorch (Python)

Configure PyTorch to use RMM for CUDA tensor allocations ({py:func}`API <rmm.allocators.torch.rmm_torch_allocator>`):

```{literalinclude} ../../python/rmm/rmm/tests/examples/guide.py
---
language: python
start-after: "# [pytorch]"
end-before: "# [/pytorch]"
dedent:
---
```

## Multi-Device Usage

For multi-GPU systems, each device can have its own memory resource. Use `set_per_device_resource_ref` (C++) or `set_per_device_resource` (Python) to configure each device before allocating memory on it:

`````{tabs}
````{group-tab} C++
```{literalinclude} ../../cpp/examples/docs/src/guide.cu
---
language: cuda
start-after: "// [multi-device]"
end-before: "// [/multi-device]"
dedent:
---
```
````
````{group-tab} Python
```{literalinclude} ../../python/rmm/rmm/tests/examples/guide.py
---
language: python
start-after: "# [multi-device]"
end-before: "# [/multi-device]"
dedent:
---
```
````
`````
