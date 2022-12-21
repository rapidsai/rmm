import rmm

if __name__ == "__main__":
    buf = rmm.DeviceBuffer(size=100)

    print(buf.size)
    print(buf.ptr)

    assert buf.size == 100
