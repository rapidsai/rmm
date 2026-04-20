# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Code examples for docs/user_guide/stream_ordered_allocation.md


def how_it_works() -> None:
    # [how-it-works]
    import rmm
    from rmm.pylibrmm.stream import Stream

    mr = rmm.mr.CudaAsyncMemoryResource()
    stream = Stream()
    buffer = rmm.DeviceBuffer(size=1000, stream=stream, mr=mr)

    # buffer.ptr is usable immediately in stream-ordered operations
    # [/how-it-works]

    assert buffer.size == 1000


def reading_results() -> None:
    # [reading-results]
    import rmm
    from rmm.pylibrmm.stream import Stream

    mr = rmm.mr.CudaAsyncMemoryResource()
    stream = Stream()
    d_buf = rmm.DeviceBuffer(size=1000, stream=stream, mr=mr)

    # ... GPU work writes to d_buf on stream ...

    # Async copy to host on the same stream, then sync before reading
    h_buf = bytearray(d_buf.size)
    d_buf.copy_to_host(h_buf, stream)
    stream.synchronize()
    # [/reading-results]


def cross_stream() -> None:
    # isort: off
    # [cross-stream]
    import rmm
    from rmm.pylibrmm.stream import Stream
    from cuda.core import Device

    dev = Device()
    dev.set_current()

    mr = rmm.mr.CudaAsyncMemoryResource()
    stream_a = dev.create_stream()
    stream_b = dev.create_stream()

    buffer = rmm.DeviceBuffer(size=1000, stream=Stream(obj=stream_a), mr=mr)

    # Record an event after the allocation on stream_a
    alloc_event = dev.create_event(options={"enable_timing": False})
    stream_a.record(alloc_event)

    # stream_b waits for the event — no CPU synchronization needed
    stream_b.wait(alloc_event)

    # Now safe to use buffer.ptr in operations on stream_b
    # [/cross-stream]
    # isort: on

    assert buffer.size == 1000


def buffer_lifetime() -> None:
    # isort: off
    # [buffer-lifetime]
    import rmm
    from rmm.pylibrmm.stream import Stream
    from cuda.core import Device

    dev = Device()
    dev.set_current()

    mr = rmm.mr.CudaAsyncMemoryResource()
    stream_a = dev.create_stream()
    stream_b = dev.create_stream()

    buffer = rmm.DeviceBuffer(size=1000, stream=Stream(obj=stream_a), mr=mr)

    # Make stream_b wait for the allocation on stream_a
    alloc_event = dev.create_event(options={"enable_timing": False})
    stream_a.record(alloc_event)
    stream_b.wait(alloc_event)

    # Use buffer on stream_b ...

    # Before destroying buffer, make stream_a wait for stream_b's work
    done_event = dev.create_event(options={"enable_timing": False})
    stream_b.record(done_event)
    stream_a.wait(done_event)

    # Now safe to destroy buffer
    del buffer
    # [/buffer-lifetime]
    # isort: on


def numba_stream_example() -> None:
    try:
        from numba import cuda
    except ImportError:
        print("Numba not available, skipping numba_stream_example")
        return

    # isort: off
    # [numba-stream]
    import rmm
    from rmm.pylibrmm.stream import Stream
    from cuda.core import Device
    from numba import cuda

    dev = Device()
    dev.set_current()

    @cuda.jit
    def kernel(data, n):
        idx = cuda.grid(1)
        if idx < n:
            data[idx] = idx * 2

    mr = rmm.mr.CudaAsyncMemoryResource()
    stream = dev.create_stream()

    buffer = rmm.DeviceBuffer(size=1000 * 4, stream=Stream(obj=stream), mr=mr)

    numba_stream = cuda.external_stream(int(stream.handle))
    kernel[100, 10, numba_stream](
        cuda.as_cuda_array(buffer).view("float32"), 1000
    )

    stream.sync()
    # [/numba-stream]
    # isort: on


if __name__ == "__main__":
    how_it_works()
    reading_results()
    cross_stream()
    buffer_lifetime()
    numba_stream_example()

    print("All stream_ordered_allocation examples passed.")
