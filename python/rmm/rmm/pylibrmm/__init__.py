# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm import memory_resource

from .device_buffer import DeviceBuffer

__all__ = ["DeviceBuffer", "memory_resource"]
