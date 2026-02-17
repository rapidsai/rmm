# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Re-export from stream for backward compatibility.
# Prefer: from rmm.pylibrmm.stream cimport CudaStream
from rmm.pylibrmm.stream cimport CudaStream
