# Copyright (c) 2025, NVIDIA CORPORATION.
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


from cuda.core.experimental import Device

import rmm.pylibrmm.stream


def test_cuda_stream_protocol():
    device = Device()
    device.set_current()
    rmm_stream = rmm.pylibrmm.stream.Stream()

    cuda_stream = device.create_stream(rmm_stream)
    assert cuda_stream is not None
