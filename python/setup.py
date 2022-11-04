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
    install_requires=[
        "cuda-python>=11.7.1,<12.0",
        "numpy>=1.19",
        "numba>=0.49",
    ],
    zip_safe=False,
)
