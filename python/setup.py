from setuptools import setup

setup(name='librmm_cffi',
      version="0.6.0",
      packages=["librmm_cffi"],
      setup_requires=["cffi>=1.0.0"],
      cffi_modules=["librmm_cffi/librmm_build.py:ffibuilder"],
      install_requires=["cffi>=1.0.0"],
      )
