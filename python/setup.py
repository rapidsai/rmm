# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

    with open(os.path.join(cuda_include_dir, "cuda.h"), "r") as f:
        for line in f.readlines():
            if re.search(r"#define CUDA_VERSION ", line) is not None:
                cuda_version = line
                break

    if cuda_version is None:
        raise TypeError("CUDA_VERSION not found in cuda.h")
    cuda_version = int(cuda_version.split()[2])
    return "%d.%d" % (cuda_version // 1000, (cuda_version % 1000) // 10)


cython_tests = glob.glob("rmm/tests/*.pyx")

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
preprocess_files = ["gpu.pxd"]
supported_cuda_versions = {"10.1", "10.2", "11.0"}

for file_p in preprocess_files:
    pxi_file = ".".join(file_p.split(".")[:-1])
    pxi_file = pxi_file + ".pxi"

    if CUDA_VERSION in supported_cuda_versions:
        shutil.copyfile(
            os.path.join(cwd, "rmm/_cuda", CUDA_VERSION, pxi_file),
            os.path.join(cwd, "rmm/_cuda", file_p),
        )
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

library_dirs = [get_python_lib(), os.path.join(os.sys.prefix, "lib")]

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
            extra_compile_args=["-std=c++14"],
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    # Include the separately-compiled shared library
    setup_requires=["cython"],
    ext_modules=extensions,
    packages=find_packages(include=["rmm", "rmm.*"]),
    package_data=dict.fromkeys(
        find_packages(include=["rmm._lib", "rmm._lib.includes", "rmm._cuda*"]),
        ["*.pxd"],
    ),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=install_requires,
    zip_safe=False,
)
