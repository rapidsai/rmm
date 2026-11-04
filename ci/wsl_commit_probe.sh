#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TEMP: WSL commit-limit probe. Measures the cudaMalloc ceiling with and without
# 12 GiB of guest RAM held by a sibling process; if Windows commit is the binding
# limit, the CUDA ceiling should drop by ~12 GiB.
set -uo pipefail
P=python
probe() {
  "$P" - "$1" <<'EOF'
import ctypes, json, sys, time
GiB = 1 << 30
rt = ctypes.CDLL("libcudart.so"); rt.cudaGetErrorName.restype = ctypes.c_char_p
rt.cudaFree(None)
f, t = ctypes.c_size_t(), ctypes.c_size_t()
rt.cudaMemGetInfo(ctypes.byref(f), ctypes.byref(t)); free0 = f.value
ptrs, err = [], 0
while True:
    p = ctypes.c_void_p()
    err = rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(GiB))
    if err: rt.cudaGetLastError(); break
    ptrs.append(p)
rt.cudaMemGetInfo(ctypes.byref(f), ctypes.byref(t))
mi = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.split(":")[0] in ("MemTotal","MemAvailable","Committed_AS","CommitLimit","SwapTotal")}
print("WSLPROBE " + json.dumps({"label": sys.argv[1], "max_gib": len(ptrs), "err": rt.cudaGetErrorName(err).decode(),
      "free0_gib": round(free0 / GiB, 2), "free_at_fail_gib": round(f.value / GiB, 2), "total_gib": round(t.value / GiB, 2), "meminfo_mib": mi}), flush=True)
for p in ptrs: rt.cudaFree(p)
EOF
}
hold() {
  "$P" - "$1" <<'EOF' &
import mmap, sys, time
n = int(float(sys.argv[1]) * (1 << 30)); m = mmap.mmap(-1, n)
for o in range(0, n, 4096): m[o] = 1
print(f"WSLPROBE holding {sys.argv[1]} GiB guest RAM", flush=True); time.sleep(900)
EOF
  echo $!
}
rapids-logger "WSL commit probe"
grep -i microsoft /proc/version || { echo "not WSL, skipping"; exit 0; }
probe baseline
H=$(hold 12); sleep 40; probe hold12; kill "$H"; wait "$H" 2>/dev/null; sleep 5
probe baseline_again
H=$(hold 20); sleep 60; probe hold20; kill "$H"; wait "$H" 2>/dev/null; sleep 5
probe baseline_final
