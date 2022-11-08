# Copyright (c) 2019-2022, NVIDIA CORPORATION.

from setuptools import find_packages
from skbuild import setup

setup(
    license_files=["LICENSE"],
    packages=find_packages(include=["rmm", "rmm.*"]),
    zip_safe=False,
)
