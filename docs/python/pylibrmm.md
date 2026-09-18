# rmm.pylibrmm

This module contains the low-level Cython bindings for RMM. Some components from this module are re-exported through the top-level `rmm` module for convenience, while others are available only through this module.

## Overview

`rmm.pylibrmm` provides the Cython layer that wraps RMM's C++ functionality:

- `DeviceBuffer` - GPU memory buffer (available as `rmm.DeviceBuffer`)
- `memory_resource` - Memory resource implementations (available as `rmm.mr`)
- Logging utilities (available through `rmm`)
- CUDA stream wrappers (documented below)

## CUDA Stream

The stream classes are available only through `rmm.pylibrmm` and provide low-level CUDA stream management.

### rmm.pylibrmm.stream

```{eval-rst}
.. automodule:: rmm.pylibrmm.stream
   :members:
   :undoc-members:
   :show-inheritance:
```

## CUDA Stream Pool

```{eval-rst}
.. automodule:: rmm.pylibrmm.cuda_stream_pool
   :members:
   :undoc-members:
   :show-inheritance:
```

## Device Buffer Functions

```{eval-rst}
.. automodule:: rmm.pylibrmm.device_buffer
   :members: copy_device_to_ptr, copy_host_to_ptr, copy_ptr_to_host, to_device
```
