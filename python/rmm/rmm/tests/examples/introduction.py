# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Code examples for docs/user_guide/introduction.md

# [basic-example]
import rmm

mr = rmm.mr.CudaAsyncMemoryResource()
buffer = rmm.DeviceBuffer(size=1024, mr=mr)
# [/basic-example]

assert buffer.size == 1024
