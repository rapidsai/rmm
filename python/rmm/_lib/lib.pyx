# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

# cython: profile = False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


# Utility Functions
def _get_error_msg(errcode):
    """
    Get error message for the given error code.
    """
    msg = rmmGetErrorString(<rmmError_t>errcode)
    cdef bytes py_msg = msg
    return py_msg.decode("utf-8")


def check_error(errcode):
    """
    Checks the error of a function that returns rmmError_t and raises a Python
    exception based on the error code
    """
    from rmm import RMMError

    if errcode != RMM_SUCCESS:
        msg = _get_error_msg(errcode)
        raise RMMError(errcode, msg)
