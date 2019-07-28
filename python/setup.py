from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import subprocess
import multiprocessing

CURRENT_DIR = os.path.dirname(os.path.normpath(os.path.abspath(__file__)))
ROOT_DIR = os.path.join(CURRENT_DIR, '..')
LIB_PATH = [os.path.join(CURRENT_DIR, 'build', 'librmm.so')]
BUILD_DIR = os.path.join(CURRENT_DIR, 'build')

def call(command):
    p = subprocess.Popen(command,
                         stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    for line in iter(p.stdout.readline, b''):
        print(line.rstrip().decode('utf-8'))

def cmake_build():
    call(['cmake', ROOT_DIR])
    call(['make', '-j', str(multiprocessing.cpu_count())])


if not os.path.exists(BUILD_DIR):
    os.mkdir(BUILD_DIR)
    os.chdir(BUILD_DIR)
    cmake_build()
    os.chdir(os.path.pardir)


setup(name='librmm_cffi',
      version="0.9.0",
      packages=["librmm_cffi"],
      setup_requires=["cffi>=1.0.0"],
      cffi_modules=["librmm_cffi/librmm_build.py:ffibuilder"],
      install_requires=["cffi>=1.0.0",
                        'numpy',
                        'numba'],
      zip_safe=False,
      data_files=[('.', LIB_PATH)])
