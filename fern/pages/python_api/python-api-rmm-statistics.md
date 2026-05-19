---
slug: api-reference/python-api-rmm-statistics
---

# rmm.statistics

Generated from RMM Python sources.

## `python/rmm/rmm/statistics.py`

### `Statistics`

```python
class Statistics
```

Statistics returned by ``{get,push,pop}_statistics()``.

Attributes
current_bytes
    Current number of bytes allocated
current_count
    Current number of allocations allocated
peak_bytes
    Peak number of bytes allocated
peak_count
    Peak number of allocations allocated
total_bytes
    Total number of bytes allocated
total_count
    Total number of allocations allocated

_Source: `python/rmm/rmm/statistics.py:24`_

### `enable_statistics`

```python
def enable_statistics()
```

Enable allocation statistics.

This function is idempotent. If statistics have been enabled for the
current RMM resource stack, this is a no-op.

Warnings
This modifies the current RMM memory resource. StatisticsResourceAdaptor
is pushed onto the current RMM memory resource stack and must remain the
topmost resource throughout the statistics gathering.

_Source: `python/rmm/rmm/statistics.py:51`_

### `get_statistics`

```python
def get_statistics()
```

Get the current allocation statistics.

Returns
If enabled, returns the current tracked statistics.
If disabled, returns None.

_Source: `python/rmm/rmm/statistics.py:71`_

### `push_statistics`

```python
def push_statistics()
```

Push new counters on the current allocation statistics stack.

This returns the current tracked statistics and pushes a new set
of zero counters on the stack of statistics.

If statistics are disabled (the current memory resource is not an
instance of StatisticsResourceAdaptor), this function is a no-op.

Returns
If enabled, returns the current tracked statistics _before_ the pop.
If disabled, returns None.

_Source: `python/rmm/rmm/statistics.py:85`_

### `pop_statistics`

```python
def pop_statistics()
```

Pop the counters of the current allocation statistics stack.

This returns the counters of current tracked statistics and pops
them from the stack.

If statistics are disabled (the current memory resource is not an
instance of StatisticsResourceAdaptor), this function is a no-op.

Returns
If enabled, returns the popped counters.
If disabled, returns None.

_Source: `python/rmm/rmm/statistics.py:105`_

### `statistics`

```python
def statistics()
```

Context to enable allocation statistics.

If statistics have been enabled already (the current memory resource is an
instance of StatisticsResourceAdaptor), new counters are pushed on the
current allocation statistics stack when entering the context and popped
again when exiting using `push_statistics()` and `pop_statistics()`.

If statistics have not been enabled, a new StatisticsResourceAdaptor is set
as the current RMM memory resource when entering the context and removed
again when exiting.

Raises
ValueError
    If the current RMM memory source was changed while in the context.

_Source: `python/rmm/rmm/statistics.py:126`_

### `ProfilerRecords`

```python
class ProfilerRecords
```

Records of the memory statistics recorded by a profiler.

_Source: `python/rmm/rmm/statistics.py:165`_

### `ProfilerContext`

```python
class ProfilerContext
```

Context manager and decorator for profiling memory usage.

This class can be used both as a decorator and as a context manager
to profile memory allocations in functions or code blocks.

records : ProfilerRecords
    The profiler records that the memory statistics are written to.
name : str
    The name of the memory profile. Mandatory when used as a context
    manager. Optional when used as a decorator.

Examples
As a decorator:

>>> @profiler()
... def my_function():
...     pass

As a context manager:

>>> with profiler(name="my_code_block"):
...     pass

_Source: `python/rmm/rmm/statistics.py:293`_

### `profiler`

```python
def profiler(records, name)
```

Decorator and context to profile function or code block.

If statistics are enabled (the current memory resource is an
instance of StatisticsResourceAdaptor), this decorator records the
memory statistics of the decorated function or code block.

If statistics are disabled, this decorator/context is a no-op.

records
    The profiler records that the memory statistics are written to. If
    not set, a default profiler records are used.
name
    The name of the memory profile, mandatory when the profiler
    is used as a context manager. If used as a decorator, an empty name
    is allowed. In this case, the name is the filename, line number, and
    function name.

_Source: `python/rmm/rmm/statistics.py:370`_
