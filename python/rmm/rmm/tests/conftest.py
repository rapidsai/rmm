import pytest

import rmm


@pytest.fixture(scope="function", autouse=True)
def rmm_auto_reinitialize():
    # Run the test
    yield

    # Automatically reinitialize the current memory resource after running each
    # test

    rmm.reinitialize()


@pytest.fixture
def stats_mr():
    mr = rmm.mr.StatisticsResourceAdaptor(rmm.mr.CudaMemoryResource())
    rmm.mr.set_current_device_resource(mr)
    return mr
