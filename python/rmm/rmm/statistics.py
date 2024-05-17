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

from contextlib import contextmanager
from typing import Dict, Optional

import rmm.mr


def enable_statistics() -> None:
    """Enable allocation statistics

    This function is idempotent, if statistics has been enabled for the
    current RMM resource stack, this is a no-op.

    Warning
    -------
    This modifies the current RMM memory resource. StatisticsResourceAdaptor
    is pushed onto the current RMM memory resource stack and must remain the
    the top must resource throughout the statistics gathering.
    """

    mr = rmm.mr.get_current_device_resource()
    if not isinstance(mr, rmm.mr.StatisticsResourceAdaptor):
        rmm.mr.set_current_device_resource(
            rmm.mr.StatisticsResourceAdaptor(mr)
        )


@contextmanager
def statistics():
    """Context to enable allocation statistics temporarily.

    Warning
    -------
    This modifies the current RMM memory resource. StatisticsResourceAdaptor
    is pushed onto the current RMM memory resource stack when entering the
    context and popped again when exiting. If statistics has been enabled for
    the current RMM resource stack already, this is a no-op.

    Raises
    ------
    ValueError
        If the RMM memory source stack was changed while in the context.
    """

    # Save the current memory resource for later cleanup
    prior_mr = rmm.mr.get_current_device_resource()
    enable_statistics()
    try:
        current_mr = rmm.mr.get_current_device_resource()
        yield
    finally:
        if current_mr is not rmm.mr.get_current_device_resource():
            raise ValueError(
                "RMM memory source stack was changed while in the context"
            )
        rmm.mr.set_current_device_resource(prior_mr)


def get_statistics() -> Optional[Dict[str, int]]:
    """Get the current allocation statistics

    Return
    ------
    If enabled, returns the current tracked statistics.
    If disabled, returns None.
    """
    mr = rmm.mr.get_current_device_resource()
    if isinstance(mr, rmm.mr.StatisticsResourceAdaptor):
        return mr.allocation_counts
    return None


def push_statistics() -> Optional[Dict[str, int]]:
    """Push new counters on the current allocation statistics stack

    This returns the current tracked statistics and push a new set
    of zero counters on the stack of statistics.

    If statistics are disabled (the current memory resource is not an
    instance of `StatisticsResourceAdaptor`), this function is a no-op.

    Return
    ------
    If enabled, returns the current tracked statistics _before_ the pop.
    If disabled, returns None.
    """
    mr = rmm.mr.get_current_device_resource()
    if isinstance(mr, rmm.mr.StatisticsResourceAdaptor):
        return mr.push_counters()
    return None


def pop_statistics() -> Optional[Dict[str, int]]:
    """Pop the counters of the current allocation statistics stack

    This returns the counters of current tracked statistics and pops
    them from the stack.

    If statistics are disabled (the current memory resource is not an
    instance of `StatisticsResourceAdaptor`), this function is a no-op.

    Return
    ------
    If enabled, returns the popped counters.
    If disabled, returns None.
    """
    mr = rmm.mr.get_current_device_resource()
    if isinstance(mr, rmm.mr.StatisticsResourceAdaptor):
        return mr.pop_counters()
    return None
