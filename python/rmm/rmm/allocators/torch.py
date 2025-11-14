# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

rmm_torch_allocator = None

try:
    from torch.cuda.memory import CUDAPluggableAllocator
except ImportError:
    pass
else:
    from torch.cuda import is_available

    if is_available():
        import pathlib

        # To support editable installs, we cannot search for the compiled torch
        # allocator .so relative to the current file because the current file
        # is pure Python and will therefore be in the source directory.
        # Instead, we search relative to an arbitrary file in the compiled
        # package. We use the librmm._logger module because it is small.
        from rmm.librmm import _logger

        sofile = pathlib.Path(_logger.__file__).parent / "_torch_allocator.so"
        rmm_torch_allocator = CUDAPluggableAllocator(
            str(sofile.absolute()),
            alloc_fn_name="allocate",
            free_fn_name="deallocate",
        )
        del pathlib, sofile
    del is_available
