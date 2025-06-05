# User Guide

Achieving optimal performance in GPU-centric workflows frequently requires
customizing how GPU ("device") memory is allocated.

RMM is a package that enables you to allocate device memory
in a highly configurable way. For example, it enables you to
allocate and use pools of GPU memory, or to use
[managed memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
for allocations.

You can also easily configure other libraries like Numba and CuPy
to use RMM for allocating device memory.

## Installation

See the project [README](https://github.com/rapidsai/rmm) for how to install RMM.

## Using RMM

There are two ways to use RMM in Python code:

1. Using the `rmm.DeviceBuffer` API to explicitly create and manage
   device memory allocations
2. Transparently via external libraries such as CuPy and Numba

RMM provides a `MemoryResource` abstraction to control _how_ device
memory is allocated in both the above uses.

### `DeviceBuffer` Objects

A `DeviceBuffer` represents an **untyped, uninitialized device memory
allocation**.  `DeviceBuffer`s can be created by providing the
size of the allocation in bytes:

```python
>>> import rmm
>>> buf = rmm.DeviceBuffer(size=100)
```

The size of the allocation and the memory address associated with it
can be accessed via the `.size` and `.ptr` attributes respectively:

```python
>>> buf.size
100
>>> buf.ptr
140202544726016
```

`DeviceBuffer`s can also be created by copying data from host memory:

```python
>>> import rmm
>>> import numpy as np
>>> a = np.array([1, 2, 3], dtype='float64')
>>> buf = rmm.DeviceBuffer.to_device(a.view("uint8"))  # to_device expects an unsigned 8-bit dtype
>>> buf.size
24
```

Conversely, the data underlying a `DeviceBuffer` can be copied to the host:

```python
>>> np.frombuffer(buf.tobytes())
array([1., 2., 3.])
```

#### Prefetching a `DeviceBuffer`

[CUDA Unified Memory](
  https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
), also known as managed memory, can be allocated using an
`rmm.mr.ManagedMemoryResource` explicitly, or by calling `rmm.reinitialize`
with `managed_memory=True`.

A `DeviceBuffer` backed by managed memory or other
migratable memory (such as
[HMM/ATS](https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management/)
memory) may be prefetched to a specified device, for example to reduce or eliminate page faults.

```python
>>> import rmm
>>> rmm.reinitialize(managed_memory=True)
>>> buf = rmm.DeviceBuffer(size=100)
>>> buf.prefetch()
```

The above example prefetches the `DeviceBuffer` memory to the current CUDA device
on the stream that the `DeviceBuffer` last used (e.g. at construction). The
destination device ID and stream are optional parameters.

```python
>>> import rmm
>>> rmm.reinitialize(managed_memory=True)
>>> from rmm.pylibrmm.stream import Stream
>>> stream = Stream()
>>> buf = rmm.DeviceBuffer(size=100, stream=stream)
>>> buf.prefetch(device=3, stream=stream) # prefetch to device on stream.
```

`DeviceBuffer.prefetch()` is a no-op if the `DeviceBuffer` is not backed
by migratable memory.

### `MemoryResource` objects

`MemoryResource` objects are used to configure how device memory allocations are made by
RMM.

By default if a `MemoryResource` is not set explicitly, RMM uses the `CudaMemoryResource`, which
uses `cudaMalloc` for allocating device memory.

`rmm.reinitialize()` provides an easy way to initialize RMM with specific memory resource options
across multiple devices. See `help(rmm.reinitialize)` for full details.

For lower-level control, the `rmm.mr.set_current_device_resource()` function can be
used to set a different MemoryResource for the current CUDA device.  For
example, enabling the `ManagedMemoryResource` tells RMM to use
`cudaMallocManaged` instead of `cudaMalloc` for allocating memory:

```python
>>> import rmm
>>> rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
```

> :warning: The default resource must be set for any device **before**
> allocating any device memory on that device.  Setting or changing the
> resource after device allocations have been made can lead to unexpected
> behaviour or crashes.

As another example, `PoolMemoryResource` allows you to allocate a
large "pool" of device memory up-front. Subsequent allocations will
draw from this pool of already allocated memory.  The example
below shows how to construct a PoolMemoryResource with an initial size
of 1 GiB and a maximum size of 4 GiB. The pool uses
`CudaMemoryResource` as its underlying ("upstream") memory resource:

```python
>>> import rmm
>>> pool = rmm.mr.PoolMemoryResource(
...     rmm.mr.CudaMemoryResource(),
...     initial_pool_size="1GiB", # equivalent to initial_pool_size=2**30
...     maximum_pool_size="4GiB"
... )
>>> rmm.mr.set_current_device_resource(pool)
```

Similarly, to use a pool of managed memory:

```python
>>> import rmm
>>> pool = rmm.mr.PoolMemoryResource(
...     rmm.mr.ManagedMemoryResource(),
...     initial_pool_size="1GiB",
...     maximum_pool_size="4GiB"
... )
>>> rmm.mr.set_current_device_resource(pool)
```

Other `MemoryResource`s include:

* `FixedSizeMemoryResource` for allocating fixed blocks of memory
* `BinningMemoryResource` for allocating blocks within specified "bin" sizes from different memory
resources

`MemoryResource`s are highly configurable and can be composed together in different ways.
See `help(rmm.mr)` for more information.

## Using RMM with third-party libraries

A number of libraries provide hooks to control their device
allocations. RMM provides implementations of these for
[CuPy](https://cupy.dev),
[numba](https://numba.readthedocs.io/en/stable/), and [PyTorch](https://pytorch.org) in the
`rmm.allocators` submodule. All these approaches configure the library
to use the _current_ RMM memory resource for device
allocations.

### Using RMM with CuPy

You can configure [CuPy](https://cupy.dev/) to use RMM for memory
allocations by setting the CuPy CUDA allocator to
`rmm.allocators.cupy.rmm_cupy_allocator`:

```python
>>> from rmm.allocators.cupy import rmm_cupy_allocator
>>> import cupy
>>> cupy.cuda.set_allocator(rmm_cupy_allocator)
```

### Using RMM with Numba

You can configure [Numba](https://numba.readthedocs.io/en/stable/) to use RMM for memory allocations using the
Numba [EMM Plugin](https://numba.readthedocs.io/en/stable/cuda/external-memory.html#setting-emm-plugin).

This can be done in two ways:

1. Setting the environment variable `NUMBA_CUDA_MEMORY_MANAGER`:

  ```bash
  $ NUMBA_CUDA_MEMORY_MANAGER=rmm.allocators.numba python (args)
  ```

2. Using the `set_memory_manager()` function provided by Numba:

  ```python
  >>> from numba import cuda
  >>> from rmm.allocators.numba import RMMNumbaManager
  >>> cuda.set_memory_manager(RMMNumbaManager)
  ```

### Using RMM with PyTorch

You can configure
[PyTorch](https://pytorch.org/docs/stable/notes/cuda.html) to use RMM
for memory allocations using their by configuring the current
allocator.

```python
>>> from rmm.allocators.torch import rmm_torch_allocator
>>> import torch

>>> torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
```

## Memory statistics and profiling

RMM can profile memory usage and track memory statistics by using either of the following:
  - Use the context manager `rmm.statistics.statistics()` to enable statistics tracking for a specific code block.
  - Call `rmm.statistics.enable_statistics()` to enable statistics tracking globally.

Common to both usages is that they modify the currently active RMM memory resource. The current device resource is wrapped with a `StatisticsResourceAdaptor` which must remain the topmost resource throughout the statistics tracking:
```python
>>> import rmm
>>> import rmm.statistics

>>> # We start with the default cuda memory resource
>>> rmm.mr.get_current_device_resource()
<rmm.pylibrmm.memory_resource.CudaMemoryResource object at 0x7fa0da48a8e0>

>>> # When using statistics, we get a StatisticsResourceAdaptor with the context
>>> with rmm.statistics.statistics():
...     rmm.mr.get_current_device_resource()
<rmm.pylibrmm.memory_resource.StatisticsResourceAdaptor object at 0x7fa0dd6e4a40>

>>> # We can also enable statistics globally
>>> rmm.statistics.enable_statistics()
>>> print(rmm.mr.get_current_device_resource())
<rmm.pylibrmm.memory_resource.StatisticsResourceAdaptor object at 0x7f9a11340a40>
```

With statistics enabled, you can query statistics of the current and peak bytes and number of allocations performed by the current RMM memory resource:
```python
>>> buf = rmm.DeviceBuffer(size=10)
>>> rmm.statistics.get_statistics()
Statistics(current_bytes=16, current_count=1, peak_bytes=16, peak_count=1, total_bytes=16, total_count=1)
```

### Memory Profiler
To profile a specific block of code, first enable memory statistics by calling `rmm.statistics.enable_statistics()`. To profile a function, use `profiler` as a function decorator:
```python
>>> @rmm.statistics.profiler()
... def f(size):
...   rmm.DeviceBuffer(size=size)
>>> f(1000)

>>> # By default, the profiler write to rmm.statistics.default_profiler_records
>>> print(rmm.statistics.default_profiler_records.report())
Memory Profiling
================

Legends:
  ncalls       - number of times the function or code block was called
  memory_peak  - peak memory allocated in function or code block (in bytes)
  memory_total - total memory allocated in function or code block (in bytes)

Ordered by: memory_peak

ncalls     memory_peak    memory_total  filename:lineno(function)
     1           1,008           1,008  <ipython-input-11-5fc63161ac29>:1(f)
```

To profile a code block, use `profiler` as a context manager:
```python
>>> with rmm.statistics.profiler(name="my code block"):
...     rmm.DeviceBuffer(size=20)
>>> print(rmm.statistics.default_profiler_records.report())
Memory Profiling
================

Legends:
  ncalls       - number of times the function or code block was called
  memory_peak  - peak memory allocated in function or code block (in bytes)
  memory_total - total memory allocated in function or code block (in bytes)

Ordered by: memory_peak

ncalls     memory_peak    memory_total  filename:lineno(function)
     1           1,008           1,008  <ipython-input-11-5fc63161ac29>:1(f)
     1              32              32  my code block
```

The `profiler` supports nesting:
```python
>>> with rmm.statistics.profiler(name="outer"):
...     buf1 = rmm.DeviceBuffer(size=10)
...     with rmm.statistics.profiler(name="inner"):
...         buf2 = rmm.DeviceBuffer(size=10)
>>> print(rmm.statistics.default_profiler_records.report())
Memory Profiling
================

Legends:
  ncalls       - number of times the function or code block was called
  memory_peak  - peak memory allocated in function or code block (in bytes)
  memory_total - total memory allocated in function or code block (in bytes)

Ordered by: memory_peak

ncalls     memory_peak    memory_total  filename:lineno(function)
     1           1,008           1,008  <ipython-input-4-865fbe04e29f>:1(f)
     1              32              32  my code block
     1              32              32  outer
     1              16              16  inner
```
