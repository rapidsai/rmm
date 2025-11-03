# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for rmm"""

from rmm.pylibrmm.stream cimport Stream


cdef Stream as_stream(Stream stream) except *:
    """
    Convert a stream argument to a Stream instance.

    This function converts the provided stream argument to a valid Stream
    instance. For now, all it does is check for None and raise a TypeError if
    the argument is None. In the future it will be extended to accept other
    types of stream representations i.e. anything supporting the
    __cuda_stream__ protocol.

    Parameters
    ----------
    stream : Stream
        The stream to convert

    Returns
    -------
    Stream
        The converted Stream instance

    Raises
    ------
    TypeError
        If stream is None
    """
    if stream is None:
        raise TypeError(
            "stream argument cannot be None. "
            "Please provide a valid Stream instance."
        )
    return stream
