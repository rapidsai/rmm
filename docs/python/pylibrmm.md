# rmm.pylibrmm

This module contains the low-level Cython bindings for RMM. The public API from this module is re-exported through the top-level `rmm` module.

## Overview

`rmm.pylibrmm` provides the Cython layer that wraps RMM's C++ functionality:

- `DeviceBuffer` - GPU memory buffer (available as `rmm.DeviceBuffer`)
- `memory_resource` - Memory resource implementations (available as `rmm.mr`)
- Logging utilities (available through `rmm`)
- CUDA stream wrappers

## Usage

Most users should access these components through the top-level `rmm` module rather than importing from `rmm.pylibrmm` directly. See the [rmm module documentation](rmm.md) for the public API.
