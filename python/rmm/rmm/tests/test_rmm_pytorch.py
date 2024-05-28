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

import gc

import pytest

from rmm.allocators.torch import rmm_torch_allocator

torch = pytest.importorskip("torch")


@pytest.fixture(scope="session")
def torch_allocator():
    try:
        from torch.cuda.memory import change_current_allocator
    except ImportError:
        pytest.skip("pytorch pluggable allocator not available")
    change_current_allocator(rmm_torch_allocator)


def test_rmm_torch_allocator(torch_allocator, stats_mr):
    assert stats_mr.allocation_counts.current_bytes == 0
    x = torch.tensor([1, 2]).cuda()
    assert stats_mr.allocation_counts.current_bytes > 0
    del x
    gc.collect()
    assert stats_mr.allocation_counts.current_bytes == 0


def test_rmm_torch_allocator_using_stream(torch_allocator, stats_mr):
    assert stats_mr.allocation_counts.current_bytes == 0
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        x = torch.tensor([1, 2]).cuda()
    torch.cuda.current_stream().wait_stream(s)
    assert stats_mr.allocation_counts.current_bytes > 0
    del x
    gc.collect()
    assert stats_mr.allocation_counts.current_bytes == 0
