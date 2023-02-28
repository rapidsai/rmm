# Copyright (c) 2019-2022, NVIDIA CORPORATION.

from setuptools import find_packages
from skbuild import setup

setup(
    packages=find_packages(include=["rmm", "rmm.*"]),
    include_package_data=True,
    zip_safe=False,
)
