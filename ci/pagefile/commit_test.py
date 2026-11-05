# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TEMP (rapidsai/rmm#2339): raise host memory use to HOST_FRACTION of the RAM visible to this
# environment, then cudaMalloc + cudaMemset all free GPU memory, sampling Windows commit along the way.
import argparse
import ctypes
import glob
import json
import mmap
import os
import platform
import subprocess
import sys
import time

GiB = 1 << 30
MiB = 1 << 20
COUNTERS = (
    "$ProgressPreference = 'SilentlyContinue'; "
    "$s = (Get-Counter '\\Memory\\Committed Bytes', '\\Memory\\Commit Limit').CounterSamples.CookedValue; "
    "$p = (Get-CimInstance Win32_PageFileUsage | Measure-Object AllocatedBaseSize -Sum).Sum; "
    "Write-Output ('{0} {1} {2}' -f $s[0], $s[1], $p)"
)


def emit(**kw):
    print("COMMIT_TEST " + json.dumps(kw), flush=True)


def windows_commit(mode):
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = (
        [
            "powershell",
            "-NoProfile",
            "-OutputFormat",
            "Text",
            "-Command",
            COUNTERS,
        ]
        if mode == "windows"
        else [sys.executable, os.path.join(here, "winps.py")]
    )
    # powershell.exe can fail to start while host memory is nearly exhausted; retry briefly.
    for _ in range(5):
        r = subprocess.run(
            cmd,
            input=None if mode == "windows" else COUNTERS,
            capture_output=True,
            text=True,
        )
        fields = r.stdout.split()[-3:]
        if len(fields) == 3:
            c, lim, pf = fields
            return {
                "committed_gib": round(float(c) / GiB, 3),
                "limit_gib": round(float(lim) / GiB, 3),
                "pagefile_mib": int(pf),
            }
        time.sleep(2)
    return {"counter_error": (r.stderr or r.stdout)[-300:]}


def host_memory():
    if platform.system() == "Windows":

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys, m.ullAvailPhys
    info = {
        ln.split(":")[0]: int(ln.split()[1]) * 1024
        for ln in open("/proc/meminfo")
    }
    return info["MemTotal"], info["MemAvailable"]


def cudart():
    pats = []
    for root in [
        *sys.path,
        os.environ.get("CONDA_PREFIX", ""),
        "/usr/local/cuda",
    ]:
        pats += [
            os.path.join(root, "nvidia", "cu13", "**", "cudart64_13.dll"),
            os.path.join(root, "nvidia", "cu13", "**", "libcudart.so.13"),
        ]
        pats += [
            os.path.join(root, "lib", "libcudart.so.13"),
            os.path.join(root, "lib64", "libcudart.so.13"),
        ]
    for pat in pats:
        hits = glob.glob(pat, recursive=True)
        if hits:
            if platform.system() == "Windows":
                os.add_dll_directory(os.path.dirname(hits[0]))
            rt = ctypes.CDLL(hits[0])
            rt.cudaGetErrorName.restype = ctypes.c_char_p
            return rt
    raise RuntimeError("libcudart not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["windows", "wsl"], required=True)
    ap.add_argument("--host-memory-fraction", type=float, default=0.95)
    args = ap.parse_args()

    total, avail = host_memory()
    emit(
        event="start",
        host_total_gib=round(total / GiB, 3),
        host_available_gib=round(avail / GiB, 3),
        **windows_commit(args.mode),
    )

    target = max(0, int(args.host_memory_fraction * total) - (total - avail))
    host = mmap.mmap(-1, target) if target else None
    for off in range(0, target, mmap.PAGESIZE):
        host[off] = 1
    total, avail = host_memory()
    emit(
        event="host_allocated",
        host_alloc_gib=round(target / GiB, 3),
        host_in_use_fraction=round((total - avail) / total, 4),
        **windows_commit(args.mode),
    )

    rt = cudart()
    rt.cudaFree(None)
    free, tot = ctypes.c_size_t(), ctypes.c_size_t()
    rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(tot))
    free0, total_gpu = free.value, tot.value
    # Stop at the reported free memory: WDDM would otherwise keep going into system-memory fallback.
    ptrs, allocated, name = [], 0, ""
    for chunk in (GiB, 64 * MiB):
        while not name and allocated + chunk <= free0:
            p = ctypes.c_void_p()
            err = rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(chunk))
            if err == 0:
                err = (
                    rt.cudaMemset(p, 1, ctypes.c_size_t(chunk))
                    or rt.cudaDeviceSynchronize()
                )
            if err:
                name = rt.cudaGetErrorName(err).decode()
                rt.cudaGetLastError()
                break
            ptrs.append(p)
            allocated += chunk
    rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(tot))
    peak = windows_commit(args.mode)
    full = not name and allocated >= free0 - 64 * MiB
    emit(
        event="gpu_allocated",
        gpu_total_gib=round(total_gpu / GiB, 3),
        gpu_free_at_start_gib=round(free0 / GiB, 3),
        gpu_allocated_gib=round(allocated / GiB, 3),
        gpu_free_at_stop_gib=round(free.value / GiB, 3),
        last_error=name,
        full_gpu=full,
        commit_headroom_gib=round(
            peak["limit_gib"] - peak["committed_gib"], 3
        ),
        **peak,
    )
    time.sleep(5)
    for p in ptrs:
        rt.cudaFree(p)
    rt.cudaDeviceSynchronize()
    if host:
        host.close()
    time.sleep(5)
    emit(event="freed", **windows_commit(args.mode))
    sys.exit(0 if full else 1)


if __name__ == "__main__":
    main()
