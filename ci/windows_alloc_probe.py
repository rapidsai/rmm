# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TEMP: cumulative cudaMalloc ceiling on native Windows, with Windows commit counters (rapidsai/rmm#2339).
import ctypes
import glob
import json
import os
import subprocess
import sys

GiB = 1 << 30


def find_cudart():
    import nvidia.cu13

    root = next(iter(nvidia.cu13.__path__))
    for d in (
        root,
        os.path.join(root, "bin"),
        os.path.join(root, "bin", "x86_64"),
    ):
        if os.path.isdir(d):
            os.add_dll_directory(d)
    hits = glob.glob(
        os.path.join(root, "**", "cudart64_*.dll"), recursive=True
    )
    return hits[0]


def commit():
    out = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-Counter '\\Memory\\Committed Bytes','\\Memory\\Commit Limit').CounterSamples.CookedValue",
        ],
        capture_output=True,
        text=True,
    ).stdout.split()
    return [round(float(x) / GiB, 2) for x in out]


def emit(**kw):
    print("WINPROBE " + json.dumps(kw), flush=True)


def main(chunk_gib):
    rt = ctypes.CDLL(find_cudart())
    rt.cudaGetErrorName.restype = ctypes.c_char_p
    rt.cudaFree(None)
    f, t = ctypes.c_size_t(), ctypes.c_size_t()
    rt.cudaMemGetInfo(ctypes.byref(f), ctypes.byref(t))
    emit(
        event="start",
        chunk_gib=chunk_gib,
        free_gib=round(f.value / GiB, 2),
        total_gib=round(t.value / GiB, 2),
        commit=commit(),
    )
    ptrs = []
    while True:
        p = ctypes.c_void_p()
        err = rt.cudaMalloc(
            ctypes.byref(p), ctypes.c_size_t(int(chunk_gib * GiB))
        )
        if err == 0:
            err = (
                rt.cudaMemset(p, 1, ctypes.c_size_t(int(chunk_gib * GiB)))
                or rt.cudaDeviceSynchronize()
            )
        if err:
            rt.cudaMemGetInfo(ctypes.byref(f), ctypes.byref(t))
            emit(
                event="fail",
                allocated_gib=len(ptrs) * chunk_gib,
                err=rt.cudaGetErrorName(err).decode(),
                free_gib=round(f.value / GiB, 2),
                commit=commit(),
            )
            break
        ptrs.append(p)
        if len(ptrs) % 8 == 0:
            emit(
                event="alloc",
                allocated_gib=len(ptrs) * chunk_gib,
                commit=commit(),
            )
    for p in ptrs:
        rt.cudaFree(p)
    rt.cudaDeviceSynchronize()
    emit(event="freed", commit=commit())


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
