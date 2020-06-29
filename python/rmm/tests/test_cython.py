import importlib

cython_test_modules = [
    "rmm.tests.test_device_buffer",
]


def test_cython():
    for mod in cython_test_modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass
