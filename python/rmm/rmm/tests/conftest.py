# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

import rmm
import rmm.statistics


@pytest.fixture(scope="function", autouse=True)
def rmm_auto_reinitialize():
    # Run the test
    yield

    # Automatically reinitialize the current memory resource after running each
    # test

    rmm.reinitialize()


@pytest.fixture
def stats_mr():
    """Fixture that makes a StatisticsResourceAdaptor available to the test"""
    with rmm.statistics.statistics():
        yield rmm.mr.get_current_device_resource()
