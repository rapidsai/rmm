# Copyright (c) 2019-2023, NVIDIA CORPORATION.

from setuptools import find_packages
from skbuild import setup

packages = find_packages(include=["rmm*"])
setup(
    packages=packages,
    package_data={key: ["VERSION", "*.pxd"] for key in packages},
    zip_safe=False,
)
