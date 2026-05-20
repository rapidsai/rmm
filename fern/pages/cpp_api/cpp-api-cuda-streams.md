---
slug: api-reference/cpp-api-cuda-streams
---

# CUDA Streams

Generated from RMM C++ headers.

## `cpp/include/rmm/cuda_stream.hpp`

### CUDA Stream Class

Owning wrapper for a CUDA stream.

Provides RAII lifetime semantics for a CUDA stream.

```cpp
class cuda_stream
```

_Source: `cpp/include/rmm/cuda_stream.hpp:29`_

### Flags Enum

stream creation flags.

```cpp
enum class flags : unsigned int
```

_Source: `cpp/include/rmm/cuda_stream.hpp:34`_

### CUDA Stream Constructor (cuda_stream.hpp:45)

Move constructor (default)

A moved-from cuda_stream is invalid and it is Undefined Behavior to call methods that access the owned stream.

```cpp
cuda_stream(cuda_stream&&) = default;
```

_Source: `cpp/include/rmm/cuda_stream.hpp:45`_

### CUDA Stream Constructor (cuda_stream.hpp:66)

Construct a new CUDA stream object

**Parameters:**

- `flags`: Stream creation flags.

**Throws:**

- `rmm::cuda_error`: if stream creation fails

```cpp
cuda_stream(cuda_stream::flags flags = cuda_stream::flags::sync_default);
```

_Source: `cpp/include/rmm/cuda_stream.hpp:66`_

### Cudastream T (cuda_stream.hpp:86)

Explicit conversion to cudaStream_t.

```cpp
explicit operator cudaStream_t() const noexcept;
```

_Source: `cpp/include/rmm/cuda_stream.hpp:86`_

### CUDA Stream View

Implicit conversion to cuda_stream_view

**Returns:** A view of the owned stream

```cpp
operator cuda_stream_view() const;
```

_Source: `cpp/include/rmm/cuda_stream.hpp:100`_

### Synchronize (cuda_stream.hpp:116)

Synchronize the owned CUDA stream.

Calls `cudaStreamSynchronize()`.

**Throws:**

- `rmm::cuda_error`: if stream synchronization fails

```cpp
void synchronize() const;
```

_Source: `cpp/include/rmm/cuda_stream.hpp:116`_

### Synchronize No Throw (cuda_stream.hpp:123)

Synchronize the owned CUDA stream. Does not throw if there is an error.

Calls `cudaStreamSynchronize()` and asserts if there is an error.

```cpp
void synchronize_no_throw() const noexcept;
```

_Source: `cpp/include/rmm/cuda_stream.hpp:123`_

## `cpp/include/rmm/cuda_stream_pool.hpp`

### CUDA Stream Pool Class

A pool of CUDA streams.

Provides efficient access to collection of CUDA stream objects.

Successive calls may return a `cuda_stream_view` of identical streams. For example, a possible implementation is to maintain a circular buffer of `cuda_stream` objects.

```cpp
class cuda_stream_pool
```

_Source: `cpp/include/rmm/cuda_stream_pool.hpp:31`_

### CUDA Stream Pool Constructor

Construct a new CUDA stream pool object of the given non-zero size

**Throws:**

- `logic_error`: if `pool_size` is zero

**Parameters:**

- `pool_size`: The number of streams in the pool
- `flags`: Flags used when creating streams in the pool.

```cpp
explicit cuda_stream_pool(std::size_t pool_size = default_size, cuda_stream::flags flags = cuda_stream::flags::sync_default);
```

_Source: `cpp/include/rmm/cuda_stream_pool.hpp:42`_

### Get Stream (cuda_stream_pool.hpp:58)

Get a `cuda_stream_view` of a stream in the pool.

This function is thread safe with respect to other calls to the same function.

**Returns:** rmm::cuda_stream_view

```cpp
rmm::cuda_stream_view get_stream() const noexcept;
```

_Source: `cpp/include/rmm/cuda_stream_pool.hpp:58`_

### Get Stream (cuda_stream_pool.hpp:73)

Get a `cuda_stream_view` of the stream associated with `stream_id`. Equivalent values of `stream_id` return a stream_view to the same underlying stream.

This function is thread safe with respect to other calls to the same function.

**Parameters:**

- `stream_id`: Unique identifier for the desired stream

**Returns:** rmm::cuda_stream_view

> **Note:** `stream_id` is wrapped around the pool size, therefore any size_t value is allowed.

```cpp
rmm::cuda_stream_view get_stream(std::size_t stream_id) const;
```

_Source: `cpp/include/rmm/cuda_stream_pool.hpp:73`_

### Get Pool Size

Get the number of streams in the pool.

This function is thread safe with respect to other calls to the same function.

**Returns:** the number of streams in the pool

```cpp
std::size_t get_pool_size() const noexcept;
```

_Source: `cpp/include/rmm/cuda_stream_pool.hpp:82`_

## `cpp/include/rmm/cuda_stream_view.hpp`

### CUDA Stream View Class

Strongly-typed non-owning wrapper for CUDA streams with default constructor.

This wrapper is simply a "view": it does not own the lifetime of the stream it wraps.

```cpp
class cuda_stream_view
```

_Source: `cpp/include/rmm/cuda_stream_view.hpp:28`_

### CUDA Stream View Constructor (cuda_stream_view.hpp:48)

Constructor from a cudaStream_t

**Parameters:**

- `stream`: The underlying stream for this view

```cpp
cuda_stream_view(cudaStream_t stream) noexcept;
```

_Source: `cpp/include/rmm/cuda_stream_view.hpp:48`_

### CUDA Stream View Constructor (cuda_stream_view.hpp:55)

Implicit conversion from stream_ref.

**Parameters:**

- `stream`: The underlying stream for this view

```cpp
cuda_stream_view(cuda::stream_ref stream) noexcept;
```

_Source: `cpp/include/rmm/cuda_stream_view.hpp:55`_

### Cudastream T (cuda_stream_view.hpp:69)

Implicit conversion to cudaStream_t.

**Returns:** cudaStream_t The underlying stream referenced by this cuda_stream_view

```cpp
operator cudaStream_t() const noexcept;
```

_Source: `cpp/include/rmm/cuda_stream_view.hpp:69`_

### Synchronize (cuda_stream_view.hpp:95)

Synchronize the viewed CUDA stream.

Calls `cudaStreamSynchronize()`.

**Throws:**

- `rmm::cuda_error`: if stream synchronization fails

```cpp
void synchronize() const;
```

_Source: `cpp/include/rmm/cuda_stream_view.hpp:95`_

### Synchronize No Throw (cuda_stream_view.hpp:102)

Synchronize the viewed CUDA stream. Does not throw if there is an error.

Calls `cudaStreamSynchronize()` and asserts if there is an error.

```cpp
void synchronize_no_throw() const noexcept;
```

_Source: `cpp/include/rmm/cuda_stream_view.hpp:102`_
