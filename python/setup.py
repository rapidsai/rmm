# Copyright (c) 2019-2022, NVIDIA CORPORATION.

import os

from setuptools import find_packages
from skbuild import setup

import versioneer

setup(
    name="rmm" + os.getenv("RAPIDS_PY_WHEEL_CUDA_SUFFIX", default=""),
    version=os.getenv(
        "RAPIDS_PY_WHEEL_VERSIONEER_OVERRIDE", default=versioneer.get_version()
    ),
    description="rmm - RAPIDS Memory Manager",
    url="https://github.com/rapidsai/rmm",
    author="NVIDIA Corporation",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    # Include the separately-compiled shared library
    extras_require={"test": ["pytest", "pytest-xdist"]},
    packages=find_packages(include=["rmm", "rmm.*"]),
    include_package_data=True,
    package_data={
        # Note: A dict comprehension with an explicit copy is necessary (rather
        # than something simpler like a dict.fromkeys) because otherwise every
        # package will refer to the same list and skbuild modifies it in place.
        key: ["*.hpp", "*.pxd"]
        for key in find_packages(
            include=["rmm._lib", "rmm._lib.includes", "rmm._cuda*"]
        )
    },
    cmdclass=versioneer.get_cmdclass(),
    install_requires=[
        "cuda-python>=11.5,<11.7.1",
        "numpy>=1.19",
        "numba>=0.49",
    ],
    zip_safe=False,
)
