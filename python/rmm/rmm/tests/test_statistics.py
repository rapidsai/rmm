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

import pytest

import rmm.mr
from rmm.statistics import (
    ProfilerRecords,
    _get_descriptive_name_of_object,
    default_profiler_records,
    get_statistics,
    pop_statistics,
    profiler,
    push_statistics,
    statistics,
)


def test_context():
    mr0 = rmm.mr.get_current_device_resource()
    assert get_statistics() is None
    with statistics():
        mr1 = rmm.mr.get_current_device_resource()
        assert isinstance(
            rmm.mr.get_current_device_resource(),
            rmm.mr.StatisticsResourceAdaptor,
        )
        b1 = rmm.DeviceBuffer(size=20)
        stats = get_statistics()
        assert stats.current_bytes == 32
        assert stats.current_count == 1
        assert stats.peak_bytes == 32
        assert stats.peak_count == 1
        assert stats.total_bytes == 32
        assert stats.total_count == 1

        with statistics():
            mr2 = rmm.mr.get_current_device_resource()
            assert mr1 is mr2
            b2 = rmm.DeviceBuffer(size=10)
            stats = get_statistics()
            assert stats.current_bytes == 16
            assert stats.current_count == 1
            assert stats.peak_bytes == 16
            assert stats.peak_count == 1
            assert stats.total_bytes == 16
            assert stats.total_count == 1

        stats = get_statistics()
        assert stats.current_bytes == 48
        assert stats.current_count == 2
        assert stats.peak_bytes == 48
        assert stats.peak_count == 2
        assert stats.total_bytes == 48
        assert stats.total_count == 2

        del b1
        del b2
    assert rmm.mr.get_current_device_resource() is mr0


def test_multiple_mr(stats_mr):
    buffers = [rmm.DeviceBuffer(size=1000) for _ in range(10)]

    for i in range(9, 0, -2):
        del buffers[i]

    stats = stats_mr.allocation_counts
    assert stats.current_bytes == 5040
    assert stats.current_count == 5
    assert stats.peak_bytes == 10080
    assert stats.peak_count == 10
    assert stats.total_bytes == 10080
    assert stats.total_count == 10

    # Push a new Tracking adaptor
    mr2 = rmm.mr.StatisticsResourceAdaptor(stats_mr)
    rmm.mr.set_current_device_resource(mr2)
    try:
        for _ in range(2):
            buffers.append(rmm.DeviceBuffer(size=1000))

        stats = mr2.allocation_counts
        assert stats.current_bytes == 2016
        assert stats.current_count == 2
        assert stats.peak_bytes == 2016
        assert stats.peak_count == 2
        assert stats.total_bytes == 2016
        assert stats.total_count == 2

        stats = stats_mr.allocation_counts
        assert stats.current_bytes == 7056
        assert stats.current_count == 7
        assert stats.peak_bytes == 10080
        assert stats.peak_count == 10
        assert stats.total_bytes == 12096
        assert stats.total_count == 12

        del buffers
        stats = mr2.allocation_counts
        assert stats.current_bytes == 0
        assert stats.current_count == 0
        assert stats.peak_bytes == 2016
        assert stats.peak_count == 2
        assert stats.total_bytes == 2016
        assert stats.total_count == 2

        stats = stats_mr.allocation_counts
        assert stats.current_bytes == 0
        assert stats.current_count == 0
        assert stats.peak_bytes == 10080
        assert stats.peak_count == 10
        assert stats.total_bytes == 12096
        assert stats.total_count == 12

    finally:
        rmm.mr.set_current_device_resource(stats_mr)


def test_counter_stack(stats_mr):
    buffers = [rmm.DeviceBuffer(size=10) for _ in range(10)]

    # push returns the stats from the top before the push
    stats = stats_mr.push_counters()  # stats from stack level 0
    assert stats.current_bytes == 160
    assert stats.current_count == 10
    assert stats.peak_bytes == 160
    assert stats.peak_count == 10
    assert stats.total_bytes == 160
    assert stats.total_count == 10

    b1 = rmm.DeviceBuffer(size=10)

    stats = stats_mr.push_counters()  # stats from stack level 1
    assert stats.current_bytes == 16
    assert stats.current_count == 1
    assert stats.peak_bytes == 16
    assert stats.peak_count == 1
    assert stats.total_bytes == 16
    assert stats.total_count == 1

    del b1

    # pop returns the popped stats
    # Note, the bytes and counts can be negative
    stats = stats_mr.pop_counters()  # stats from stack level 2
    assert stats.current_bytes == -16
    assert stats.current_count == -1
    assert stats.peak_bytes == 0
    assert stats.peak_count == 0
    assert stats.total_bytes == 0
    assert stats.total_count == 0

    b1 = rmm.DeviceBuffer(size=10)

    stats = stats_mr.push_counters()  # stats from stack level 1
    assert stats.current_bytes == 16
    assert stats.current_count == 1
    assert stats.peak_bytes == 16
    assert stats.peak_count == 1
    assert stats.total_bytes == 32
    assert stats.total_count == 2

    b2 = rmm.DeviceBuffer(size=10)

    stats = stats_mr.pop_counters()  # stats from stack level 2
    assert stats.current_bytes == 16
    assert stats.current_count == 1
    assert stats.peak_bytes == 16
    assert stats.peak_count == 1
    assert stats.total_bytes == 16
    assert stats.total_count == 1

    stats = stats_mr.pop_counters()  # stats from stack level 1
    assert stats.current_bytes == 32
    assert stats.current_count == 2
    assert stats.peak_bytes == 32
    assert stats.peak_count == 2
    assert stats.total_bytes == 48
    assert stats.total_count == 3

    del b1
    del b2

    stats = stats_mr.allocation_counts  # stats from stack level 0
    assert stats.current_bytes == 160
    assert stats.current_count == 10
    assert stats.peak_bytes == 192
    assert stats.peak_count == 12
    assert stats.total_bytes == 208
    assert stats.total_count == 13

    del buffers
    with pytest.raises(IndexError, match="cannot pop the last counter pair"):
        stats_mr.pop_counters()


def test_current_statistics(stats_mr):
    b1 = rmm.DeviceBuffer(size=10)
    stats = get_statistics()
    assert stats.current_bytes == 16
    assert stats.current_count == 1
    assert stats.peak_bytes == 16
    assert stats.peak_count == 1
    assert stats.total_bytes == 16
    assert stats.total_count == 1

    b2 = rmm.DeviceBuffer(size=20)
    stats = push_statistics()
    assert stats.current_bytes == 48
    assert stats.current_count == 2
    assert stats.peak_bytes == 48
    assert stats.peak_count == 2
    assert stats.total_bytes == 48
    assert stats.total_count == 2

    del b1
    stats = pop_statistics()
    assert stats.current_bytes == -16
    assert stats.current_count == -1
    assert stats.peak_bytes == 0
    assert stats.peak_count == 0
    assert stats.total_bytes == 0
    assert stats.total_count == 0

    del b2
    stats = get_statistics()
    assert stats.current_bytes == 0
    assert stats.current_count == 0
    assert stats.peak_bytes == 48
    assert stats.peak_count == 2
    assert stats.total_bytes == 48
    assert stats.total_count == 2


def test_statistics_disabled():
    assert get_statistics() is None
    assert push_statistics() is None
    assert get_statistics() is None


def test_profiler(stats_mr):
    profiler_records = ProfilerRecords()
    assert len(profiler_records.records) == 0
    assert "No data" in profiler_records.report()

    @profiler(records=profiler_records)
    def f1():
        b1 = rmm.DeviceBuffer(size=10)
        b2 = rmm.DeviceBuffer(size=10)
        del b1
        return b2

    b1 = f1()
    b2 = f1()

    @profiler(records=profiler_records)
    def f2():
        b1 = rmm.DeviceBuffer(size=10)

        @profiler(records=profiler_records, name="g2")
        def g2(b1):
            b2 = rmm.DeviceBuffer(size=10)
            del b1
            return b2

        return g2(b1)

    f2()
    f2()
    del b1
    del b2
    f2()

    @profiler(records=profiler_records)
    def f3():
        return [rmm.DeviceBuffer(size=100) for _ in range(100)]

    f3()

    records = profiler_records.records
    assert records[
        _get_descriptive_name_of_object(f1)
    ] == ProfilerRecords.MemoryRecord(
        num_calls=2, memory_total=64, memory_peak=32
    )
    assert records[
        _get_descriptive_name_of_object(f2)
    ] == ProfilerRecords.MemoryRecord(
        num_calls=3, memory_total=96, memory_peak=32
    )
    assert records["g2"] == ProfilerRecords.MemoryRecord(
        num_calls=3, memory_total=48, memory_peak=16
    )
    assert records[
        _get_descriptive_name_of_object(f3)
    ] == ProfilerRecords.MemoryRecord(
        num_calls=1, memory_total=11200, memory_peak=11200
    )

    @profiler()  # use the default profiler records
    def f4():
        return [rmm.DeviceBuffer(size=10) for _ in range(10)]

    f4()

    with profiler(name="b1 and b2"):  # use the profiler as a context manager
        b1 = rmm.DeviceBuffer(size=100)
        b2 = rmm.DeviceBuffer(size=100)
        with profiler(name="del b1 and b2"):
            del b1
            del b2

    records = default_profiler_records.records
    assert records[
        _get_descriptive_name_of_object(f4)
    ] == ProfilerRecords.MemoryRecord(
        num_calls=1, memory_total=160, memory_peak=160
    )
    assert records["b1 and b2"] == ProfilerRecords.MemoryRecord(
        num_calls=1, memory_total=224, memory_peak=224
    )
    assert records["del b1 and b2"] == ProfilerRecords.MemoryRecord(
        num_calls=1, memory_total=0, memory_peak=0
    )
