# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Code examples for docs/user_guide/installation.md


def test_installation() -> None:
    # [test-installation]
    import rmm

    print(rmm.__version__)

    # Quick test
    buffer = rmm.DeviceBuffer(size=100)
    print(f"Allocated {buffer.size} bytes")
    # [/test-installation]

    assert buffer.size == 100


if __name__ == "__main__":
    test_installation()

    print("All installation examples passed.")
