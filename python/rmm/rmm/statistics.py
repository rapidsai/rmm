# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Literal

import rmm.mr


@dataclass
class Statistics:
    """Statistics returned by ``{get,push,pop}_statistics()``.

    Attributes
    ----------
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
    """

    current_bytes: int
    current_count: int
    peak_bytes: int
    peak_count: int
    total_bytes: int
    total_count: int


def enable_statistics() -> None:
    """Enable allocation statistics.

    This function is idempotent. If statistics have been enabled for the
    current RMM resource stack, this is a no-op.

    Warnings
    --------
    This modifies the current RMM memory resource. StatisticsResourceAdaptor
    is pushed onto the current RMM memory resource stack and must remain the
    topmost resource throughout the statistics gathering.
    """

    mr = rmm.mr.get_current_device_resource()
    if not isinstance(mr, rmm.mr.StatisticsResourceAdaptor):
        rmm.mr.set_current_device_resource(
            rmm.mr.StatisticsResourceAdaptor(mr)
        )


def get_statistics() -> Statistics | None:
    """Get the current allocation statistics.

    Returns
    -------
    If enabled, returns the current tracked statistics.
    If disabled, returns None.
    """
    mr = rmm.mr.get_current_device_resource()
    if isinstance(mr, rmm.mr.StatisticsResourceAdaptor):
        return mr.allocation_counts
    return None


def push_statistics() -> Statistics | None:
    """Push new counters on the current allocation statistics stack.

    This returns the current tracked statistics and pushes a new set
    of zero counters on the stack of statistics.

    If statistics are disabled (the current memory resource is not an
    instance of StatisticsResourceAdaptor), this function is a no-op.

    Returns
    -------
    If enabled, returns the current tracked statistics _before_ the pop.
    If disabled, returns None.
    """
    mr = rmm.mr.get_current_device_resource()
    if isinstance(mr, rmm.mr.StatisticsResourceAdaptor):
        return mr.push_counters()
    return None


def pop_statistics() -> Statistics | None:
    """Pop the counters of the current allocation statistics stack.

    This returns the counters of current tracked statistics and pops
    them from the stack.

    If statistics are disabled (the current memory resource is not an
    instance of StatisticsResourceAdaptor), this function is a no-op.

    Returns
    -------
    If enabled, returns the popped counters.
    If disabled, returns None.
    """
    mr = rmm.mr.get_current_device_resource()
    if isinstance(mr, rmm.mr.StatisticsResourceAdaptor):
        return mr.pop_counters()
    return None


@contextmanager
def statistics():
    """Context to enable allocation statistics.

    If statistics have been enabled already (the current memory resource is an
    instance of StatisticsResourceAdaptor), new counters are pushed on the
    current allocation statistics stack when entering the context and popped
    again when exiting using `push_statistics()` and `push_statistics()`.

    If statistics have not been enabled, a new StatisticsResourceAdaptor is set
    as the current RMM memory resource when entering the context and removed
    again when exiting.

    Raises
    ------
    ValueError
        If the current RMM memory source was changed while in the context.
    """

    prior_non_stats_mr = None
    if push_statistics() is None:
        # Save the current non-statistics memory resource for later cleanup
        prior_non_stats_mr = rmm.mr.get_current_device_resource()
        enable_statistics()

    try:
        current_mr = rmm.mr.get_current_device_resource()
        yield
    finally:
        if current_mr is not rmm.mr.get_current_device_resource():
            raise ValueError(
                "RMM memory source stack was changed "
                "while in the statistics context"
            )
        if prior_non_stats_mr is None:
            pop_statistics()
        else:
            rmm.mr.set_current_device_resource(prior_non_stats_mr)


class ProfilerRecords:
    """Records of the memory statistics recorded by a profiler."""

    @dataclass
    class MemoryRecord:
        """Memory statistics of a single code block.

        Attributes
        ----------
        num_calls
            Number of times this code block was invoked.
        memory_total
            Total number of bytes allocated.
        memory_peak
            Peak number of bytes allocated.
        """

        num_calls: int = 0
        memory_total: int = 0
        memory_peak: int = 0

        def add(self, memory_total: int, memory_peak: int):
            self.num_calls += 1
            self.memory_total += memory_total
            self.memory_peak = max(self.memory_peak, memory_peak)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, ProfilerRecords.MemoryRecord] = defaultdict(
            ProfilerRecords.MemoryRecord
        )

    def add(self, name: str, data: Statistics) -> None:
        """Add memory statistics to the record named `name`.

        This method is thread-safe.

        Parameters
        ----------
        name
            Name of the record.
        data
            Memory statistics of `name`.
        """
        with self._lock:
            self._records[name].add(
                memory_total=data.total_bytes, memory_peak=data.peak_bytes
            )

    @property
    def records(self) -> dict[str, MemoryRecord]:
        """Dictionary mapping record names to their memory statistics."""
        return dict(self._records)

    def report(
        self,
        ordered_by: Literal[
            "num_calls", "memory_peak", "memory_total"
        ] = "memory_peak",
    ) -> str:
        """Pretty format the recorded memory statistics.

        Parameters
        ----------
        ordered_by
            Sort the statistics by this attribute.

        Returns
        -------
        The pretty formatted string of the memory statistics
        """

        # Sort by `ordered_by`
        records = sorted(
            ((name, data) for name, data in self.records.items()),
            key=lambda x: getattr(x[1], ordered_by),
            reverse=True,
        )
        ret = "Memory Profiling\n"
        ret += "================\n\n"
        if len(records) == 0:
            return ret + "No data, maybe profiling wasn't enabled?"
        ret += (
            "Legends:\n"
            "  ncalls       - number of times the function or code block "
            "was called\n"
            "  memory_peak  - peak memory allocated in function or code "
            "block (in bytes)\n"
            "  memory_total - total memory allocated in function or code "
            "block (in bytes)\n"
        )
        ret += f"\nOrdered by: {ordered_by}\n"
        ret += "\nncalls     memory_peak    memory_total  "
        ret += "filename:lineno(function)\n"
        for name, data in records:
            ret += f"{data.num_calls:6,d} {data.memory_peak:15,d} "
            ret += f"{data.memory_total:15,d}  {name}\n"
        return ret[:-1]  # Remove the final newline

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.records})"

    def __str__(self) -> str:
        return self.report()


def _get_descriptive_name_of_object(obj: object) -> str:
    """Get descriptive name of object.

    Parameters
    ----------
    obj
        Object in question

    Returns
    -------
    A string including filename, line number, and object name.
    """

    obj = inspect.unwrap(obj)
    _, linenumber = inspect.getsourcelines(obj)
    filepath = inspect.getfile(obj)
    return f"{filepath}:{linenumber}({obj.__qualname__})"


default_profiler_records = ProfilerRecords()


def profiler(
    *,
    records: ProfilerRecords = default_profiler_records,
    name: str = "",
):
    """Decorator and context to profile function or code block.

    If statistics are enabled (the current memory resource is an
    instance of StatisticsResourceAdaptor), this decorator records the
    memory statistics of the decorated function or code block.

    If statistics are disabled, this decorator/context is a no-op.

    Parameters
    ----------
    records
        The profiler records that the memory statistics are written to. If
        not set, a default profiler records are used.
    name
        The name of the memory profile, mandatory when the profiler
        is used as a context manager. If used as a decorator, an empty name
        is allowed. In this case, the name is the filename, line number, and
        function name.
    """

    class ProfilerContext:
        def __call__(self, func: callable) -> callable:
            _name = name or _get_descriptive_name_of_object(func)

            @wraps(func)
            def wrapper(*args, **kwargs):
                push_statistics()
                try:
                    return func(*args, **kwargs)
                finally:
                    if (stats := pop_statistics()) is not None:
                        records.add(name=_name, data=stats)

            return wrapper

        def __enter__(self):
            if not name:
                raise ValueError(
                    "When profiler is used as a context manager, "
                    "a name must be provided"
                )
            push_statistics()
            return self

        def __exit__(self, *exc):
            if (stats := pop_statistics()) is not None:
                records.add(name=name, data=stats)
            return False

    return ProfilerContext()
