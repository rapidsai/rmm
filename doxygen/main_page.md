# librmm

Achieving optimal performance in GPU-centric workflows frequently requires customizing how host and
device memory are allocated. For example, using "pinned" host memory for asynchronous
host <-> device memory transfers, or using a device memory pool sub-allocator to reduce the cost of
dynamic device memory allocation.

The goal of the RAPIDS Memory Manager (RMM) is to provide:
- A common interface that allows customizing device and host memory allocation
- A collection of implementations of the interface
- A collection of data structures that use the interface for memory allocation

\htmlonly For more information on APIs provided by rmm, see <a href="modules.html">the modules page</a>\endhtmlonly.
