#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export SCCACHE_IDLE_TIMEOUT="0"
export SCCACHE_ERROR_LOG="/tmp/sccache.log"
export SCCACHE_SERVER_LOG="sccache=debug"
export SCCACHE_DIST_REQUEST_TIMEOUT="7140"
# shellcheck disable=SC2155
export SCCACHE_DIST_SCHEDULER_URL="https://$(if test "$(uname -m)" = x86_64; then echo amd64; else echo arm64; fi).linux.sccache.rapids.nvidia.com"
export SCCACHE_DIST_AUTH_TYPE="token"
export SCCACHE_DIST_AUTH_TOKEN="$GH_TOKEN"
export SCCACHE_DIST_MAX_RETRIES="inf"
export SCCACHE_DIST_FALLBACK_TO_LOCAL_COMPILE="false"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS }-t=100"

# Install rapidsai/sccache
wget --no-hsts -q -O- https://api.github.com/repos/rapidsai/sccache/releases/latest \
  | jq -r ".assets[] | select(.name | test(\"^sccache-v.*?-$(uname -m)-unknown-linux-musl.tar.gz\$\")) | .browser_download_url" \
  | wget --no-hsts -q -O- -i- \
  | tar -C /usr/bin -zf - --wildcards --strip-components=1 -x '*/sccache' 2>/dev/null

sccache --start-server
sccache --dist-status | jq -r '["scheduler URL: " + .SchedulerStatus[0], "server count: " + (.SchedulerStatus[1].servers | length | tostring)][]' || true

ulimit -n "$(ulimit -Hn)"
# shellcheck disable=SC2155
export PARALLEL_LEVEL="$(ulimit -Hn)"
echo "nofile ulimit: $(ulimit -n):$(ulimit -Hn)"
