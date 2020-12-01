# Copyright (c) 2019-2020, NVIDIA CORPORATION.
from setuptools import find_packages
from skbuild import setup

import versioneer

setup(
    name="rmm",
    version="0.18.0",
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(include=["rmm", "rmm.*"]),
    package_data=dict.fromkeys(
        find_packages(include=["rmm._lib", "rmm._lib.includes", "rmm._cuda*"]),
        ["*.pxd"],
    ),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
