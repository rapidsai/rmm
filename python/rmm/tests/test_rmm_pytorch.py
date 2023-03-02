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
    assert stats_mr.allocation_counts["current_bytes"] == 0
    x = torch.tensor([1, 2]).cuda()
    assert stats_mr.allocation_counts["current_bytes"] > 0
    del x
    gc.collect()
    assert stats_mr.allocation_counts["current_bytes"] == 0


def test_rmm_torch_allocator_using_stream(torch_allocator, stats_mr):
    assert stats_mr.allocation_counts["current_bytes"] == 0
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        x = torch.tensor([1, 2]).cuda()
    torch.cuda.current_stream().wait_stream(s)
    assert stats_mr.allocation_counts["current_bytes"] > 0
    del x
    gc.collect()
    assert stats_mr.allocation_counts["current_bytes"] == 0
