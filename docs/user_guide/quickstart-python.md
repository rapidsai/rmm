# Python Quick Start

## Basic Usage

```python
import rmm

# RMM initializes with CudaMemoryResource by default
# For better performance, use the async memory resource
rmm.reinitialize(pool_allocator=False)  # Uses CudaAsyncMemoryResource

# Allocate device memory
buffer = rmm.DeviceBuffer(size=1024)  # 1024 bytes

# Get pointer and size
print(f"Allocated {buffer.size} bytes at {hex(buffer.ptr)}")

# Copy data from host to device
import numpy as np
host_data = np.array([1, 2, 3, 4], dtype=np.float32)
buffer = rmm.DeviceBuffer.to_device(host_data.view('uint8'))

# Copy data back to host
host_copy = np.frombuffer(buffer.tobytes(), dtype=np.float32)
print(host_copy)  # [1. 2. 3. 4.]
```

## Using a Pool Allocator

```python
import rmm

# Create a memory pool with 1 GiB initial size
rmm.reinitialize(
    pool_allocator=True,
    initial_pool_size=2**30  # 1 GiB
)

# All allocations now use the pool
buffer = rmm.DeviceBuffer(size=1024)
```

## Configuring Memory Resources

```python
import rmm

# Option 1: Use rmm.reinitialize (simple)
rmm.reinitialize(
    pool_allocator=False,  # Use async MR (recommended)
    managed_memory=False,  # Don't use managed memory
    devices=[0]  # Configure device 0
)

# Option 2: Set memory resource directly (more control)
mr = rmm.mr.CudaAsyncMemoryResource()
rmm.mr.set_current_device_resource(mr)

# Option 3: Use a pool wrapping async MR
pool = rmm.mr.PoolMemoryResource(
    rmm.mr.CudaAsyncMemoryResource(),
    initial_pool_size=2**30,  # 1 GiB
    maximum_pool_size=2**32   # 4 GiB
)
rmm.mr.set_current_device_resource(pool)
```

## Integration with CuPy

```python
import rmm
import cupy as cp
from rmm.allocators.cupy import rmm_cupy_allocator

# Configure RMM
rmm.reinitialize(pool_allocator=False)

# Set CuPy to use RMM
cp.cuda.set_allocator(rmm_cupy_allocator)

# All CuPy arrays now use RMM
array = cp.zeros(1000)
result = cp.sqrt(array)
```

## Integration with Numba

```python
from numba import cuda
from rmm.allocators.numba import RMMNumbaManager
import rmm

# Configure RMM
rmm.reinitialize(pool_allocator=False)

# Set Numba to use RMM
cuda.set_memory_manager(RMMNumbaManager)

# Numba device arrays now use RMM
@cuda.jit
def kernel(x):
    idx = cuda.grid(1)
    if idx < x.size:
        x[idx] = idx * 2

x = cuda.device_array(100)
kernel[10, 10](x)
```

Alternatively, use the environment variable:

```bash
NUMBA_CUDA_MEMORY_MANAGER=rmm.allocators.numba python script.py
```

## Integration with PyTorch

```python
import rmm
import torch
from rmm.allocators.torch import rmm_torch_allocator

# Configure RMM
rmm.reinitialize(pool_allocator=False)

# Set PyTorch to use RMM
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

# All PyTorch tensors now use RMM
tensor = torch.zeros(1000, device='cuda')
result = tensor + 1
```

## Memory Statistics

```python
import rmm

# Enable statistics tracking
rmm.statistics.enable_statistics()

# Run some allocations
buffer1 = rmm.DeviceBuffer(size=1000)
buffer2 = rmm.DeviceBuffer(size=2000)

# Get statistics
stats = rmm.statistics.get_statistics()
print(f"Current bytes: {stats.current_bytes}")
print(f"Peak bytes: {stats.peak_bytes}")
print(f"Total allocations: {stats.total_count}")

# Or use context manager
with rmm.statistics.statistics():
    buffer = rmm.DeviceBuffer(size=5000)
    stats = rmm.statistics.get_statistics()
    print(f"Allocated: {stats.current_bytes} bytes")
```

## Memory Profiling

```python
import rmm

# Enable statistics first
rmm.statistics.enable_statistics()

# Profile a function
@rmm.statistics.profiler()
def my_function(size):
    return rmm.DeviceBuffer(size=size)

# Run the function
my_function(10000)

# View profiling report
print(rmm.statistics.default_profiler_records.report())
```
