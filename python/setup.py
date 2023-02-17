# Copyright (c) 2019-2022, NVIDIA CORPORATION.

from setuptools import find_packages
from skbuild import setup

setup(
    name="rmm",
    version="23.04.00",
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
        "Programming Language :: Python :: 3.10",
    ],
    # Include the separately-compiled shared library
    extras_require={"test": ["pytest"]},
    packages=find_packages(include=["rmm", "rmm.*"]),
    include_package_data=True,
    python_requires=">=3.8",
    package_data={
        # Note: A dict comprehension with an explicit copy is necessary (rather
        # than something simpler like a dict.fromkeys) because otherwise every
        # package will refer to the same list and skbuild modifies it in place.
        key: ["*.hpp", "*.pxd"]
        for key in find_packages(
            include=["rmm._lib", "rmm._lib.includes", "rmm._cuda*"]
        )
    },
    install_requires=[
        "cuda-python>=11.7.1,<12.0",
        "numpy>=1.19",
        "numba>=0.49",
    ],
    zip_safe=False,
)
