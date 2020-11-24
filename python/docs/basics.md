# RMM - the RAPIDS Memory Manager

Stable and nightly documentation can be found in [RAPIDS Docs](https://docs.rapids.ai/api/rmm/nightly/)

Achieving optimal performance in GPU-centric workflows frequently requires
customizing how GPU ("device") memory is allocated.

RMM is a package that enables you to allocate device memory
in a highly configurable way. For example, it enables you to
allocate and use pools of GPU memory, or to use
[managed memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
for allocations.

You can also easily configure other libraries like Numba and CuPy
to use RMM for allocating device memory.

## Installation

See the project [README](https://github.com/rapidsai/rmm) for how to install RMM.

