# Memory Resources

RMM memory resources are allocator objects that control where and how memory is allocated.

Memory resources implement allocation and deallocation for a kind of memory, for example CUDA device memory, managed memory, or pinned host memory. Resource adaptors wrap another memory resource and change its behavior, such as adding logging, tracking, limits, alignment, or prefetching, while delegating the actual allocation to the wrapped upstream resource.

Use the pages below to browse the core memory resource implementations and the adaptors that add behavior to an upstream resource.

```{toctree}
:maxdepth: 1

memory_resources
memory_resource_adaptors
```
