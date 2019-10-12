#
# Copyright (c) 2019, NVIDIA CORPORATION.
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
#

import pytest

import numpy as np

from numba import cuda
import rmm

test_num_rows = [1, 10, 1000]
test_num_cols = [1, 10, 1000]


@pytest.mark.parametrize('num_rows', test_num_rows)
@pytest.mark.parametrize('num_cols', test_num_cols)
def test_numba_gc(num_rows, num_cols):
    rand_mat = (np.random.rand(1000, 10)*10).astype(np.float32)
    X = cuda.to_device(rand_mat)
    X_rmm = rmm.to_device(X)

    X = X.transpose()
    X_rmm = X_rmm.transpose()

    np.testing.assert_equal(X.copy_to_host(), X_rmm.copy_to_host())
