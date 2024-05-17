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
from rmm.statistics import statistics


@pytest.fixture
def stats_mr():
    with statistics():
        yield rmm.mr.get_current_device_resource()


def test_context():
    prior_mr = rmm.mr.get_current_device_resource()
    with statistics():
        assert isinstance(
            rmm.mr.get_current_device_resource(),
            rmm.mr.StatisticsResourceAdaptor,
        )
    assert rmm.mr.get_current_device_resource() is prior_mr


def test_multiple_mr(stats_mr):
    buffers = [rmm.DeviceBuffer(size=1000) for _ in range(10)]

    for i in range(9, 0, -2):
        del buffers[i]

    assert stats_mr.allocation_counts == {
        "current_bytes": 5040,
        "current_count": 5,
        "peak_bytes": 10080,
        "peak_count": 10,
        "total_bytes": 10080,
        "total_count": 10,
    }

    # Push a new Tracking adaptor
    mr2 = rmm.mr.StatisticsResourceAdaptor(stats_mr)
    rmm.mr.set_current_device_resource(mr2)

    for _ in range(2):
        buffers.append(rmm.DeviceBuffer(size=1000))

    assert mr2.allocation_counts == {
        "current_bytes": 2016,
        "current_count": 2,
        "peak_bytes": 2016,
        "peak_count": 2,
        "total_bytes": 2016,
        "total_count": 2,
    }
    assert stats_mr.allocation_counts == {
        "current_bytes": 7056,
        "current_count": 7,
        "peak_bytes": 10080,
        "peak_count": 10,
        "total_bytes": 12096,
        "total_count": 12,
    }

    del buffers

    assert mr2.allocation_counts == {
        "current_bytes": 0,
        "current_count": 0,
        "peak_bytes": 2016,
        "peak_count": 2,
        "total_bytes": 2016,
        "total_count": 2,
    }
    assert stats_mr.allocation_counts == {
        "current_bytes": 0,
        "current_count": 0,
        "peak_bytes": 10080,
        "peak_count": 10,
        "total_bytes": 12096,
        "total_count": 12,
    }
    rmm.mr.set_current_device_resource(stats_mr)


def test_counter_stack(stats_mr):
    buffers = [rmm.DeviceBuffer(size=10) for _ in range(10)]

    # push returns the stats from the top before the push
    assert stats_mr.push_counters() == {  # stats from stack level 0
        "current_bytes": 160,
        "current_count": 10,
        "peak_bytes": 160,
        "peak_count": 10,
        "total_bytes": 160,
        "total_count": 10,
    }
    b1 = rmm.DeviceBuffer(size=10)
    assert stats_mr.push_counters() == {  # stats from stack level 1
        "current_bytes": 16,
        "current_count": 1,
        "peak_bytes": 16,
        "peak_count": 1,
        "total_bytes": 16,
        "total_count": 1,
    }
    del b1
    # pop returns the stats from the top before the pop.
    # Note, the bytes and counts can be negative
    assert stats_mr.pop_counters() == {  # stats from stack level 2
        "current_bytes": -16,
        "current_count": -1,
        "peak_bytes": 0,
        "peak_count": 0,
        "total_bytes": 0,
        "total_count": 0,
    }
    b1 = rmm.DeviceBuffer(size=10)
    assert stats_mr.push_counters() == {  # stats from stack level 1
        "current_bytes": 16,
        "current_count": 1,
        "peak_bytes": 16,
        "peak_count": 1,
        "total_bytes": 32,
        "total_count": 2,
    }
    b2 = rmm.DeviceBuffer(size=10)
    assert stats_mr.pop_counters() == {  # stats from stack level 2
        "current_bytes": 16,
        "current_count": 1,
        "peak_bytes": 16,
        "peak_count": 1,
        "total_bytes": 16,
        "total_count": 1,
    }
    assert stats_mr.pop_counters() == {  # stats from stack level 1
        "current_bytes": 32,
        "current_count": 2,
        "peak_bytes": 32,
        "peak_count": 2,
        "total_bytes": 48,
        "total_count": 3,
    }
    del b1
    del b2
    assert stats_mr.allocation_counts == {  # stats from stack level 0
        "current_bytes": 160,
        "current_count": 10,
        "peak_bytes": 192,
        "peak_count": 12,
        "total_bytes": 208,
        "total_count": 13,
    }
    del buffers
    with pytest.raises(IndexError, match="cannot pop the last counter pair"):
        stats_mr.pop_counters()
