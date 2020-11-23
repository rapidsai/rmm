# Copyright (c) 2020, NVIDIA CORPORATION.
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

import os

from libc.stdint cimport uintptr_t


cpdef enum cudaStream:
    cudaStreamDefault = 0
    cudaStreamLegacy = 1
    cudaStreamPerThread = 2


cdef class CudaStreamView:

    def __cinit__(self, cudaStream stream=cudaStreamDefault):
        """Construct a view of the specified CUDA stream

        Parameters
        ----------
        stream : uintptr_t, optional
            CUDA stream to wrap, default 0
        """
        if (stream == cudaStreamDefault):
            if int(os.environ.get("RMM_PER_THREAD_DEFAULT_STREAM", "0")) != 0:
                self.c_obj.reset(new cuda_stream_view(<cudaStream_t>cudaStreamPerThread))
            else:
                self.c_obj.reset(new cuda_stream_view())
        else:
            self.c_obj.reset(new cuda_stream_view(<cudaStream_t>stream))

    cpdef bool is_default(self) except *:
        """Returns True if this is the CUDA default stream
        """
        return self.c_obj.get()[0].is_default()

    cpdef bool is_per_thread_default(self) except *:
        """Returns True if this is a CUDA per-thread default stream
        """
        return self.c_obj.get()[0].is_per_thread_default()
