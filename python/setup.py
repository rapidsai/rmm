# Copyright (c) 2019-2023, NVIDIA CORPORATION.

from setuptools import find_packages
from skbuild import setup

setup(
    packages=find_packages(include=["rmm*"]),
    zip_safe=False,
)
