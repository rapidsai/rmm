# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RMM logging functionality."""

import os
import warnings

import pytest
from test_helpers import _allocs, _dtypes, _nelems, array_tester

import rmm
from rmm.pylibrmm.logger import level_enum
from rmm.pylibrmm.memory_resource._memory_resource import _flush_logs


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_rmm_csv_log(dtype, nelem, alloc, tmpdir):
    suffix = ".csv"

    base_name = str(tmpdir.join("rmm_log.csv"))
    rmm.reinitialize(logging=True, log_file_name=base_name)
    array_tester(dtype, nelem, alloc)
    _flush_logs()

    # Need to open separately because the device ID is appended to filename
    fname = base_name[: -len(suffix)] + ".dev0" + suffix
    try:
        with open(fname, "rb") as f:
            csv = f.read()
            assert csv.find(b"Time,Action,Pointer,Size,Stream") >= 0
    finally:
        os.remove(fname)


@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("nelem", _nelems)
@pytest.mark.parametrize("alloc", _allocs)
def test_rmm_enable_disable_logging(dtype, nelem, alloc, tmpdir):
    suffix = ".csv"

    base_name = str(tmpdir.join("rmm_log.csv"))

    rmm.enable_logging(log_file_name=base_name)
    print(rmm.mr.get_per_device_resource(0))
    array_tester(dtype, nelem, alloc)
    _flush_logs()

    # Need to open separately because the device ID is appended to filename
    fname = base_name[: -len(suffix)] + ".dev0" + suffix
    try:
        with open(fname, "rb") as f:
            csv = f.read()
            assert csv.find(b"Time,Action,Pointer,Size,Stream") >= 0
    finally:
        os.remove(fname)

    rmm.disable_logging()


@pytest.mark.parametrize("level", level_enum)
def test_valid_logging_level(level):
    default_level = level_enum.info
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="RMM will not log level_enum.trace."
        )
        warnings.filterwarnings(
            "ignore", message="RMM will not log level_enum.debug."
        )
        rmm.set_logging_level(level)
        assert rmm.get_logging_level() == level
        rmm.set_logging_level(default_level)  # reset to default

        rmm.set_flush_level(level)
        assert rmm.get_flush_level() == level
        rmm.set_flush_level(default_level)  # reset to default

        rmm.should_log(level)


@pytest.mark.parametrize(
    "level", ["INFO", 3, "invalid", 100, None, 1.2345, [1, 2, 3]]
)
def test_invalid_logging_level(level):
    with pytest.raises(TypeError):
        rmm.set_logging_level(level)
    with pytest.raises(TypeError):
        rmm.set_flush_level(level)
    with pytest.raises(TypeError):
        rmm.should_log(level)
