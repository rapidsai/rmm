#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Check GPU status"
nvidia-smi

rapids-logger "Check memory configuration"
nvidia-smi -q | grep "Addressing Mode" || echo "Addressing Mode not reported"
grep Coherent /proc/driver/nvidia/params || echo "Coherent GPU memory mode not reported"
grep -o 'init_on_alloc=[^ ]*' /proc/cmdline || echo "init_on_alloc: kernel default"
