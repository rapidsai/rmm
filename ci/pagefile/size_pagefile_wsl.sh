#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TEMP (rapidsai/rmm#2339): measure the WSL VM's memory here, then size the Windows host's pagefile.
set -euo pipefail
here="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
mem_total=$(awk '/^MemTotal:/ {print $2 * 1024}' /proc/meminfo)
mem_free=$(awk '/^MemFree:/ {print $2 * 1024}' /proc/meminfo)
used=$((mem_total - mem_free))
{
  echo '& {'
  cat "${here}/size_pagefile.ps1"
  echo "} -Mode wsl -WslMemTotalBytes ${mem_total} -WslUsedBytes ${used} $*"
} | python "${here}/winps.py"
