# rmm.pylibrmm

This module contains the low-level Cython bindings for RMM. Some components from this module are re-exported through the top-level `rmm` module for convenience, while others are available only through this module.

## Overview

`rmm.pylibrmm` provides the Cython layer that wraps RMM's C++ functionality:

- `DeviceBuffer` - GPU memory buffer (available as `rmm.DeviceBuffer`)
- `memory_resource` - Memory resource implementations (available as `rmm.mr`)
- Logging utilities (available through `rmm`)
- CUDA stream wrappers (documented below)

## CUDA Stream Classes

The stream classes are available only through `rmm.pylibrmm` and provide low-level CUDA stream management.

### rmm.pylibrmm.stream

```{eval-rst}
.. automodule:: rmm.pylibrmm.stream
   :members:
   :undoc-members:
   :show-inheritance:
```

### rmm.pylibrmm.cuda_stream

```{eval-rst}
.. automodule:: rmm.pylibrmm.cuda_stream
   :members:
   :undoc-members:
   :show-inheritance:
```
