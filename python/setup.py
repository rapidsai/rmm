# Copyright (c) 2019-2022, NVIDIA CORPORATION.

from setuptools import find_packages
from skbuild import setup

import versioneer

# TODO: Need to find a way to enable builds without debug or async like we had
# before with a custom cmdclass.
setup(
    name="rmm",
    version=versioneer.get_version(),
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
    # Include the separately-compiled shared library
    extras_require={"test": ["pytest", "pytest-xdist"]},
    packages=find_packages(include=["rmm", "rmm.*"]),
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
    install_requires=["numba", "cython", "cuda-python"],
    zip_safe=False,
)
