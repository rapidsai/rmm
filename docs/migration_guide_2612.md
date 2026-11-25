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

## Explicit Streams for `exec_policy` and `device_buffer`

### `exec_policy` and `exec_policy_nosync` Require a Stream

The `rmm::exec_policy` and `rmm::exec_policy_nosync` constructors no longer
default to the default stream. Pass the stream explicitly:

```cpp
#include <rmm/exec_policy.hpp>

#include <cuda/stream>

// Before
auto policy = rmm::exec_policy();

// After
auto policy = rmm::exec_policy(stream);
```

To keep the previous behavior, pass
`cuda::stream_ref{cudaStream_t{cudaStreamDefault}}`.

### Empty `device_buffer` Has an Invalid Stream

A default-constructed `rmm::device_buffer`, and a `device_buffer` that has
been moved from, now return an invalid stream
(`cuda::stream_ref{cuda::invalid_stream_t{}}`) from `stream()` instead of the
default stream. Specify a stream, for example with `set_stream()`, before
relying on `stream()` for an empty buffer.
