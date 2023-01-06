import rmm

if __name__ == "__main__":
    buf = rmm.DeviceBuffer(size=100)
    assert buf.size == 100
