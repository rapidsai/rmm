# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for rmm"""

from rmm.pylibrmm.stream cimport Stream


cdef Stream as_stream(Stream stream) except *:
    """
    Convert a stream argument to a Stream instance.

    This function checks if the provided stream is None and raises a TypeError
    if it is. This helps catch programming errors where None is accidentally
    passed as a stream argument in Cython code, which would otherwise be
    allowed by Cython's type system.

    Parameters
    ----------
    stream : Stream
        The stream to convert

    Returns
    -------
    Stream
        The input stream if it is not None

    Raises
    ------
    TypeError
        If stream is None

    Notes
    -----
    This function is designed to allow for future enhancements to support
    other stream input types beyond the Stream class.
    """
    if stream is None:
        raise TypeError(
            "stream argument cannot be None. "
            "Please provide a valid Stream instance."
        )
    return stream
