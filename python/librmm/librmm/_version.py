# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import importlib.resources

__version__ = (
    importlib.resources.files(__package__)
    .joinpath("VERSION")
    .read_text()
    .strip()
)
__git_commit__ = ""

__all__ = ["__git_commit__", "__version__"]
