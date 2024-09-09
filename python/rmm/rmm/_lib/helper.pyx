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


cdef dict BYTE_SIZES = {
    'b': 1,
    '': 1,
    'kb': 1000,
    'mb': 1000**2,
    'gb': 1000**3,
    'tb': 1000**4,
    'pb': 1000**5,
    'kib': 1024,
    'mib': 1024**2,
    'gib': 1024**3,
    'tib': 1024**4,
    'pib': 1024**5,
}


pattern = re.compile(r"^([0-9]+(?:\.[0-9]*)?)[\t ]*((?i:(?:[kmgtp]i?)?b))?$")

cdef object parse_bytes(object s):
    """Parse a string or integer into a number of bytes.

    Parameters
    ----------
    s : int | str
        Size in bytes. If an integer is provided, it is returned as-is.
        A string is parsed as a floating point number with an (optional,
        case-insensitive) byte-specifier, both SI prefixes (kb, mb, ..., pb)
        and binary prefixes (kib, mib, ..., pib) are supported.

     Returns
     -------
     Requested size in bytes as an integer.

     Raises
     ------
     ValueError
         If it is not possible to parse the input as a byte specification.
    """
    cdef str suffix
    cdef double n
    cdef int multiplier

    if isinstance(s, int):
        return s

    match = pattern.match(s)

    if match is None:
        raise ValueError(f"Could not parse {s} as a byte specification")

    n = float(match.group(1))

    suffix = match.group(2)
    if suffix is None:
        suffix = ""

    multiplier = BYTE_SIZES[suffix.lower()]

    return int(n*multiplier)
