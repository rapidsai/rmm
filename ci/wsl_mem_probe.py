# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import glob
import os

lib = sorted(
    glob.glob(os.path.join(os.environ["CONDA_PREFIX"], "lib", "libcudart.so*"))
)[0]
rt = ctypes.CDLL(lib)
GiB = 1 << 30


def info():
    f, t = ctypes.c_size_t(), ctypes.c_size_t()
    assert rt.cudaMemGetInfo(ctypes.byref(f), ctypes.byref(t)) == 0
    return f.value, t.value


def malloc(n):
    p = ctypes.c_void_p()
    err = rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n))
    if err:
        rt.cudaGetLastError()
        return None
    return p


def largest(hi):
    lo, best = 0, 0
    step = 256 << 20
    while hi - lo > step:
        mid = (lo + hi) // 2 // 256 * 256
        p = malloc(mid)
        if p:
            rt.cudaFree(p)
            lo = best = mid
        else:
            hi = mid
    return best


def show(tag):
    f, t = info()
    print(
        f"{tag}: free={f} ({f / GiB:.2f} GiB) total={t} ({t / GiB:.2f} GiB)",
        flush=True,
    )
    return f


f0 = show("initial")
print(f"largest single cudaMalloc: {largest(f0) / GiB:.2f} GiB", flush=True)
show("after search")
a = malloc(f0 // 2)
print(f"first half-of-free {f0 // 2 / GiB:.2f} GiB ok={bool(a)}", flush=True)
f1 = show("after first")
p = malloc(f1 // 2)
print(f"second half-of-free {f1 // 2 / GiB:.2f} GiB ok={bool(p)}", flush=True)
if p:
    rt.cudaFree(p)
print(
    f"largest second cudaMalloc after first: {largest(f1) / GiB:.2f} GiB",
    flush=True,
)
rt.cudaFree(a)
show("after free")

chunks = []
while (c := malloc(GiB)) is not None:
    chunks.append(c)
print(f"cumulative 1 GiB chunks: {len(chunks)} GiB", flush=True)
show("with all chunks held")
for c in chunks:
    rt.cudaFree(c)
show("after freeing chunks")
for pct in (90, 85, 80, 75, 70, 60, 50):
    n = f0 * pct // 100 // 256 * 256
    p = malloc(n)
    print(
        f"single {pct}% of free = {n} B ({n / GiB:.2f} GiB) ok={bool(p)}",
        flush=True,
    )
    if p:
        rt.cudaFree(p)
