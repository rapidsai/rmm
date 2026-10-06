# RMM 26.12 Migration Guide

## Removed CUDA Stream View API

RMM 26.12 removes the deprecated `rmm::cuda_stream_view` class and
`<rmm/cuda_stream_view.hpp>` header. Use `cuda::stream_ref` from `<cuda/stream>`
instead. Replace calls to the view's `.value()` and `.synchronize()` methods
with `.get()` and `.sync()`, respectively.

### Views of Owning Streams

`rmm::cuda_stream::view()` and the conversion to `rmm::cuda_stream_view` are
removed. Use the existing implicit conversion to `cuda::stream_ref`:

```cpp
#include <rmm/cuda_stream.hpp>

#include <cuda/stream>

rmm::cuda_stream stream;
cuda::stream_ref ref = stream;
```

### Default Stream Constants

The deprecated stream constants are also removed:

| Removed API | Replacement |
| --- | --- |
| `rmm::cuda_stream_view{}` or `rmm::cuda_stream_default` | `cuda::stream_ref{cudaStream_t{cudaStreamDefault}}` |
| `rmm::cuda_stream_legacy` | `cuda::stream_ref{cudaStreamLegacy}` |
| `rmm::cuda_stream_per_thread` | `cuda::stream_ref{cudaStreamPerThread}` |

### Cython Declarations

`rmm.librmm.cuda_stream_view` is removed. Import the replacement type from
`rmm.librmm.cuda_stream_ref` and update declarations to use `stream_ref`:

```cython
from rmm.librmm.cuda_stream_ref cimport stream_ref
```
