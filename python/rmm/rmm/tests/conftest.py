# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator

import pytest

import rmm
import rmm.mr
import rmm.statistics


@pytest.fixture(scope="function", autouse=True)
def rmm_auto_reinitialize() -> Generator[None, None, None]:
    # Run the test
    yield

    # Automatically reinitialize the current memory resource after running each
    # test

    rmm.reinitialize()


@pytest.fixture
def stats_mr() -> Generator[rmm.mr.DeviceMemoryResource, None, None]:
    """Fixture that makes a StatisticsResourceAdaptor available to the test"""
    with rmm.statistics.statistics():
        yield rmm.mr.get_current_device_resource()
