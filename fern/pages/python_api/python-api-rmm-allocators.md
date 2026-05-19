---
slug: api-reference/python-api-rmm-allocators
---

# rmm.allocators

Generated from RMM Python sources.

## `python/rmm/rmm/allocators/cupy.py`

### `rmm_cupy_allocator`

```python
def rmm_cupy_allocator(nbytes)
```

A CuPy allocator that makes use of RMM.

Examples
>>> from rmm.allocators.cupy import rmm_cupy_allocator
>>> import cupy
>>> cupy.cuda.set_allocator(rmm_cupy_allocator)

_Source: `python/rmm/rmm/allocators/cupy.py:14`_

## `python/rmm/rmm/allocators/numba.py`

### `RMMNumbaManager`

```python
class RMMNumbaManager
```

External Memory Management Plugin implementation for Numba. Provides
on-device allocation only.

See https://numba.readthedocs.io/en/stable/cuda/external-memory.html for
details of the interface being implemented here.

_Source: `python/rmm/rmm/allocators/numba.py:48`_
