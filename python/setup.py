# Copyright (c) 2019-2021, NVIDIA CORPORATION.

import filecmp
import glob
import os
import re
import shutil
import sysconfig
from distutils.sysconfig import get_python_lib

from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

import versioneer

install_requires = ["numba", "cython"]


def get_cuda_version_from_header(cuda_include_dir):

    cuda_version = None

    with open(
        os.path.join(cuda_include_dir, "cuda.h"), "r", encoding="utf-8"
    ) as f:
        for line in f.readlines():
            if re.search(r"#define CUDA_VERSION ", line) is not None:
                cuda_version = line
                break

    if cuda_version is None:
        raise TypeError("CUDA_VERSION not found in cuda.h")
    cuda_version = int(cuda_version.split()[2])
    return "%d.%d" % (cuda_version // 1000, (cuda_version % 1000) // 10)


cython_tests = glob.glob("rmm/_lib/tests/*.pyx")

CUDA_HOME = os.environ.get("CUDA_HOME", False)
if not CUDA_HOME:
    path_to_cuda_gdb = shutil.which("cuda-gdb")
    if path_to_cuda_gdb is None:
        raise OSError(
            "Could not locate CUDA. "
            "Please set the environment variable "
            "CUDA_HOME to the path to the CUDA installation "
            "and try again."
        )
    CUDA_HOME = os.path.dirname(os.path.dirname(path_to_cuda_gdb))

if not os.path.isdir(CUDA_HOME):
    raise OSError(f"Invalid CUDA_HOME: directory does not exist: {CUDA_HOME}")

cuda_include_dir = os.path.join(CUDA_HOME, "include")
cuda_lib_dir = os.path.join(CUDA_HOME, "lib64")
CUDA_VERSION = get_cuda_version_from_header(cuda_include_dir)

INSTALL_PREFIX = os.environ.get("INSTALL_PREFIX", False)
if os.path.isdir(INSTALL_PREFIX):
    rmm_include_dir = os.path.join(INSTALL_PREFIX, "include")
else:
    # use uninstalled headers in source tree
    rmm_include_dir = "../include"

# Preprocessor step to specify correct pxd file with
# valid symbols for specific version of CUDA.

cwd = os.getcwd()
files_to_preprocess = ["gpu.pxd"]

# The .pxi file is unchanged between some CUDA versions
# (e.g., 11.0 & 11.1), so we keep only a single copy
# of it
cuda_version_to_pxi_dir = {
    "10.1": "10.1",
    "10.2": "10.2",
    "11.0": "11.x",
    "11.1": "11.x",
    "11.2": "11.x",
}

for pxd_basename in files_to_preprocess:
    pxi_basename = os.path.splitext(pxd_basename)[0] + ".pxi"
    if CUDA_VERSION in cuda_version_to_pxi_dir:
        pxi_pathname = os.path.join(
            cwd,
            "rmm/_cuda",
            cuda_version_to_pxi_dir[CUDA_VERSION],
            pxi_basename,
        )
        pxd_pathname = os.path.join(cwd, "rmm/_cuda", pxd_basename)
        try:
            if filecmp.cmp(pxi_pathname, pxd_pathname):
                # files are the same, no need to copy
                continue
        except FileNotFoundError:
            # pxd_pathname doesn't exist yet
            pass
        shutil.copyfile(pxi_pathname, pxd_pathname)
    else:
        raise TypeError(f"{CUDA_VERSION} is not supported.")

try:
    nthreads = int(os.environ.get("PARALLEL_LEVEL", "0") or "0")
except Exception:
    nthreads = 0

include_dirs = [
    rmm_include_dir,
    os.path.dirname(sysconfig.get_path("include")),
    cuda_include_dir,
]

library_dirs = [
    get_python_lib(),
    os.path.join(os.sys.prefix, "lib"),
    cuda_lib_dir,
]

# lib:
extensions = cythonize(
    [
        Extension(
            "*",
            sources=["rmm/_lib/*.pyx"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            runtime_library_dirs=[
                cuda_lib_dir,
                os.path.join(os.sys.prefix, "lib"),
            ],
            libraries=["cuda", "cudart"],
            language="c++",
            extra_compile_args=["-std=c++17"],
        )
    ],
    nthreads=nthreads,
    compiler_directives=dict(
        profile=False, language_level=3, embedsignature=True,
    ),
)


# cuda:
extensions += cythonize(
    [
        Extension(
            "*",
            sources=["rmm/_cuda/*.pyx"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            runtime_library_dirs=[
                cuda_lib_dir,
                os.path.join(os.sys.prefix, "lib"),
            ],
            libraries=["cuda", "cudart"],
            language="c++",
            extra_compile_args=["-std=c++14"],
        )
    ],
    nthreads=nthreads,
    compiler_directives=dict(
        profile=False, language_level=3, embedsignature=True,
    ),
)

# tests:
extensions += cythonize(
    [
        Extension(
            "*",
            sources=cython_tests,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            runtime_library_dirs=[
                cuda_lib_dir,
                os.path.join(os.sys.prefix, "lib"),
            ],
            libraries=["cuda", "cudart"],
            language="c++",
            extra_compile_args=["-std=c++14"],
        )
    ],
    nthreads=nthreads,
    compiler_directives=dict(
        profile=True, language_level=3, embedsignature=True, binding=True
    ),
)

setup(
    name="rmm",
    version="21.08.00",
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
    setup_requires=["Cython>=0.29,<0.30"],
    extras_require={"test": ["pytest", "pytest-xdist"]},
    ext_modules=extensions,
    packages=find_packages(include=["rmm", "rmm.*"]),
    package_data=dict.fromkeys(
        find_packages(include=["rmm._lib", "rmm._lib.includes", "rmm._cuda*"]),
        ["*.hpp", "*.pxd"],
    ),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=install_requires,
    zip_safe=False,
)
