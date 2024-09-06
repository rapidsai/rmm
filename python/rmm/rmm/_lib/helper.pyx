# Copyright (c) 2024, NVIDIA CORPORATION.
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

"""Helper functions for rmm"""

import re

from libc.stdlib cimport atof


cdef dict BYTE_SIZES = {
    'b': 1,
    '': 1,
    'kb': 1000,
    'mb': 1000000,
    'gb': 1000000000,
    'tb': 1000000000000,
    'pb': 1000000000000000,
    'k': 1000,
    'm': 1000000,
    'g': 1000000000,
    't': 1000000000000,
    'p': 1000000000000000,
    'kib': 1024,
    'mib': 1048576,
    'gib': 1073741824,
    'tib': 1099511627776,
    'pib': 1125899906842624,
    'ki': 1024,
    'mi': 1048576,
    'gi': 1073741824,
    'ti': 1099511627776,
    'pi': 1125899906842624,
}


cdef object parse_bytes(object s):
    """ Parse byte string to numbers

    Parameters
    ----------
    s : int | str
        Size in bytes
    """
    cdef str suffix
    cdef double n
    cdef int multiplier

    if isinstance(s, int):
        return int(s)

    pattern = re.compile(r"^([0-9]+(?:\.[0-9]*)?)[\t ]*((?i:(?:[kmgtp]i?)?b)?)$")
    match = pattern.match(s)

    if match is None:
        raise ValueError(f"Could not parse {s} as a byte specification")

    n = atof(match.group(1).encode())

    suffix = match.group(2)
    if suffix is None:
        suffix = ""

    multiplier = BYTE_SIZES[suffix.lower()]

    return int(n*multiplier)
