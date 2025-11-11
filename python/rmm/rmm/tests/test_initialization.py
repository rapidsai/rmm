# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RMM initialization and reinitialization."""

import numpy as np
import pytest

import rmm


def test_reinitialize_initial_pool_size_gt_max():
    with pytest.raises(RuntimeError) as e:
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=1 << 11,
            maximum_pool_size=1 << 10,
        )
    assert "Initial pool size exceeds the maximum pool size" in str(e.value)


def test_reinitialize_with_valid_str_arg_pool_size():
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size="2kib",
        maximum_pool_size="8kib",
    )


def test_reinitialize_with_invalid_str_arg_pool_size():
    with pytest.raises(ValueError) as e:
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size="2k",  # 2kb valid, not 2k
            maximum_pool_size="8k",
        )
    assert "Could not parse" in str(e.value)


@pytest.fixture
def make_reinit_hook():
    funcs = []

    def _make_reinit_hook(func, *args, **kwargs):
        funcs.append(func)
        rmm.register_reinitialize_hook(func, *args, **kwargs)
        return func

    yield _make_reinit_hook
    for func in funcs:
        rmm.unregister_reinitialize_hook(func)


def test_reinit_hooks_register(make_reinit_hook):
    L = []
    make_reinit_hook(lambda: L.append(1))
    make_reinit_hook(lambda: L.append(2))
    make_reinit_hook(lambda x: L.append(x), 3)

    rmm.reinitialize()
    assert L == [3, 2, 1]


def test_reinit_hooks_unregister(make_reinit_hook):
    L = []
    one = make_reinit_hook(lambda: L.append(1))
    make_reinit_hook(lambda: L.append(2))

    rmm.unregister_reinitialize_hook(one)
    rmm.reinitialize()
    assert L == [2]


def test_reinit_hooks_register_twice(make_reinit_hook):
    L = []

    def func_with_arg(x):
        L.append(x)

    def func_without_arg():
        L.append(2)

    make_reinit_hook(func_with_arg, 1)
    make_reinit_hook(func_without_arg)
    make_reinit_hook(func_with_arg, 3)
    make_reinit_hook(func_without_arg)

    rmm.reinitialize()
    assert L == [2, 3, 2, 1]


def test_reinit_hooks_unregister_twice_registered(make_reinit_hook):
    # unregistering a twice-registered function
    # should unregister both instances:
    L = []

    def func_with_arg(x):
        L.append(x)

    make_reinit_hook(func_with_arg, 1)
    make_reinit_hook(lambda: L.append(2))
    make_reinit_hook(func_with_arg, 3)

    rmm.unregister_reinitialize_hook(func_with_arg)
    rmm.reinitialize()
    assert L == [2]


def test_available_device_memory():
    from rmm.mr import available_device_memory

    initial_memory = available_device_memory()
    device_buffer = rmm.DeviceBuffer.to_device(  # noqa: F841
        np.zeros(10000, dtype="u1")
    )
    final_memory = available_device_memory()
    assert initial_memory[1] == final_memory[1]
    assert initial_memory[0] > 0
    assert final_memory[0] > 0
