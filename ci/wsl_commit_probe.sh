#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TEMP: WSL device-memory ceiling probe (rapidsai/rmm#2339).
set -uo pipefail

grep -i microsoft /proc/version || { echo "not WSL, skipping"; exit 0; }

rapids-logger "WSL probe: environment"
echo "WSLPROBE_ENV kernel=$(uname -r)"
nvidia-smi -q > /tmp/nvsmi-q.txt 2>&1 || true
grep -nE "Driver Version|CUDA Version|Product Name|Driver Model|Current|Addressing Mode|EGM|C2C|Virtualization Mode|Total|Reserved|BAR1|Compute Mode" /tmp/nvsmi-q.txt | head -60
echo "----- full nvidia-smi -q -----"; cat /tmp/nvsmi-q.txt; echo "----- end -----"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.reserved,memory.used,memory.free,addressing_mode --format=csv 2>&1
grep -E "MemTotal|MemAvailable|SwapTotal|SwapFree|CommitLimit|Committed_AS" /proc/meminfo
awk '{print $5}' /proc/self/mountinfo | grep -E "wsl|drivers|libcuda" | head -20
stat -c "%n -> %N %s" /usr/lib/x86_64-linux-gnu/libcuda.so* /usr/lib/wsl/lib/* 2>&1 | head -20
md5sum /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/wsl/drivers/*/libcuda.so.1.1 /usr/lib/wsl/drivers/*/libcuda_loader.so 2>/dev/null
stat -c "%n" /dev/dxg /dev/nvidia* 2>&1 | head
df -h /dev/shm | tail -1
nproc

rapids-logger "WSL probe: allocations"
PROBE_PY=$(mktemp --suffix=.py)
cat > "${PROBE_PY}" <<'EOF'
import ctypes, glob, json, mmap, multiprocessing as mp, os, time
GiB = 1 << 30
LIB = (sorted(glob.glob(os.path.join(os.environ.get("CONDA_PREFIX", "/nonexistent"), "lib", "libcudart.so.*")))
       + sorted(glob.glob("/usr/local/cuda/lib64/libcudart.so.*")))[0]


def emit(**kw):
    print("WSLPROBE " + json.dumps(kw), flush=True)


def rt():
    r = ctypes.CDLL(LIB)
    r.cudaGetErrorName.restype = ctypes.c_char_p
    r.cudaFree(None)
    return r


def meminfo(r):
    f, t = ctypes.c_size_t(), ctypes.c_size_t()
    r.cudaMemGetInfo(ctypes.byref(f), ctypes.byref(t))
    return f.value, t.value


def guest():
    keys = ("MemAvailable", "Committed_AS")
    return {l.split(":")[0]: int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.split(":")[0] in keys}


def chunked(label, chunk=GiB, touch=False, api="cudaMalloc"):
    r = rt()
    free0, total = meminfo(r)
    ptrs, n, name, t0 = [], 0, "", time.time()
    while True:
        p = ctypes.c_void_p()
        if api == "cudaMalloc":
            err = r.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(chunk))
        elif api == "cudaMallocAsync":
            err = r.cudaMallocAsync(ctypes.byref(p), ctypes.c_size_t(chunk), None) or r.cudaStreamSynchronize(None)
        else:
            err = r.cudaMallocManaged(ctypes.byref(p), ctypes.c_size_t(chunk), 1)
        if err == 0 and touch:
            err = r.cudaMemset(p, 1, ctypes.c_size_t(chunk)) or r.cudaDeviceSynchronize()
        if err:
            name = r.cudaGetErrorName(err).decode()
            r.cudaGetLastError()
            break
        ptrs.append(p)
        n += 1
        if n * chunk > 400 * GiB:
            name = "cap"
            break
    f, _ = meminfo(r)
    emit(label=label, api=api, touch=touch, chunk_gib=chunk / GiB, max_gib=n * chunk / GiB, err=name,
         free0_gib=round(free0 / GiB, 2), free_at_stop_gib=round(f / GiB, 2), total_gib=round(total / GiB, 2),
         secs=round(time.time() - t0, 1), guest_mib=guest())
    for p in ptrs:
        if api == "cudaMallocAsync":
            r.cudaFreeAsync(p, None)
        else:
            r.cudaFree(p)
    r.cudaDeviceSynchronize()


def single(label):
    r = rt()
    free, _ = meminfo(r)
    lo, hi = 0, free
    while hi - lo > GiB // 4:
        mid = (lo + hi) // 2
        p = ctypes.c_void_p()
        if r.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(mid)) == 0:
            r.cudaFree(p)
            lo = mid
        else:
            r.cudaGetLastError()
            hi = mid
    emit(label=label, max_single_gib=round(lo / GiB, 2), free_gib=round(free / GiB, 2), pct_of_free=round(100 * lo / free, 1))


def run(fn, label, **k):
    p = mp.get_context("spawn").Process(target=fn, args=(label,), kwargs=k)
    p.start()
    p.join(1800)
    if p.exitcode != 0:
        emit(label=label, exitcode=p.exitcode)


def hold_guest(gib, ev):
    n = int(gib * GiB)
    m = mmap.mmap(-1, n)
    for o in range(0, n, 4096):
        m[o] = 1
    ev.set()
    time.sleep(3600)


def hold_cuda(gib, ev):
    r = rt()
    p = ctypes.c_void_p()
    err = r.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(int(gib * GiB)))
    emit(label="holder_cuda", gib=gib, err=err)
    ev.set()
    time.sleep(3600)


def with_holder(target, gib, label, **k):
    ctx = mp.get_context("spawn")
    ev = ctx.Event()
    h = ctx.Process(target=target, args=(gib, ev))
    h.start()
    ev.wait(600)
    time.sleep(5)
    emit(label=label + "_holder_ready", guest_mib=guest())
    run(chunked, label, **k)
    h.terminate()
    h.join()
    time.sleep(5)


if __name__ == "__main__":
    run(single, "single_bisect")
    run(chunked, "chunk1g")
    run(chunked, "chunk1g_touch", touch=True)
    run(chunked, "chunk4g", chunk=4 * GiB)
    run(chunked, "async1g", api="cudaMallocAsync")
    run(chunked, "managed1g_touch", api="cudaMallocManaged", touch=True)
    with_holder(hold_guest, 12, "chunk1g_guest_hold12")
    with_holder(hold_cuda, 20, "chunk1g_other_proc_holds20")
    run(chunked, "chunk1g_final")
EOF
python "${PROBE_PY}"
