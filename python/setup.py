# Copyright (c) 2019-2021, NVIDIA CORPORATION.

import filecmp
import glob
import os
import re
import shutil
import sysconfig

# Must import in this order:
#   setuptools -> Cython.Distutils.build_ext -> setuptools.command.build_ext
# Otherwise, setuptools.command.build_ext ends up inheriting from
# Cython.Distutils.old_build_ext which we do not want
import setuptools

try:
    from Cython.Distutils.build_ext import new_build_ext as _build_ext
except ImportError:
    from setuptools.command.build_ext import build_ext as _build_ext

from distutils.sysconfig import get_python_lib

import setuptools.command.build_ext
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
    "11": "11.x",
}

for pxd_basename in files_to_preprocess:
    pxi_basename = os.path.splitext(pxd_basename)[0] + ".pxi"
    pxi_dir = cuda_version_to_pxi_dir.get(CUDA_VERSION)
    if not pxi_dir:
        # didn't get an exact match on major.minor version - see if
        # we have a match on just the major version
        pxi_dir = cuda_version_to_pxi_dir.get(CUDA_VERSION.split(".")[0])

    if pxi_dir:
        pxi_pathname = os.path.join(cwd, "rmm/_cuda", pxi_dir, pxi_basename,)
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

extensions = [
    # lib:
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
    ),
    # cuda:
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
        extra_compile_args=["-std=c++17"],
    ),
    # tests:
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
        extra_compile_args=["-std=c++17"],
    ),
]


def remove_flags(compiler, *flags):
    for flag in flags:
        try:
            compiler.compiler_so = list(
                filter((flag).__ne__, compiler.compiler_so)
            )
        except Exception:
            pass


class build_ext_no_debug(_build_ext):
    def build_extensions(self):
        # Full optimization
        self.compiler.compiler_so.append("-O3")
        # No debug symbols, full optimization, no '-Wstrict-prototypes' warning
        remove_flags(
            self.compiler, "-g", "-G", "-O1", "-O2", "-Wstrict-prototypes"
        )
        super().build_extensions()

    def finalize_options(self):
        if self.distribution.ext_modules:
            # Delay import this to allow for Cython-less installs
            from Cython.Build.Dependencies import cythonize

            nthreads = getattr(self, "parallel", None)  # -j option in Py3.5+
            nthreads = int(nthreads) if nthreads else None
            self.distribution.ext_modules = cythonize(
                self.distribution.ext_modules,
                nthreads=nthreads,
                force=self.force,
                gdb_debug=False,
                compiler_directives=dict(
                    profile=False,
                    language_level=3,
                    embedsignature=True,
                    binding=True,
                ),
            )
        # Skip calling super() and jump straight to setuptools
        setuptools.command.build_ext.build_ext.finalize_options(self)


cmdclass = dict()
cmdclass.update(versioneer.get_cmdclass())
cmdclass["build_ext"] = build_ext_no_debug

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
    setup_requires=["Cython>=0.29,<0.30"],
    extras_require={"test": ["pytest", "pytest-xdist"]},
    ext_modules=extensions,
    packages=find_packages(include=["rmm", "rmm.*"]),
    package_data=dict.fromkeys(
        find_packages(include=["rmm._lib", "rmm._lib.includes", "rmm._cuda*"]),
        ["*.hpp", "*.pxd"],
    ),
    cmdclass=cmdclass,
    install_requires=install_requires,
    zip_safe=False,
)
