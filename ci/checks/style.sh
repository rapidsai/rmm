#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
#####################
# RMM Style Tester #
#####################

# Ignore errors and set path
set +e
PATH=/opt/conda/bin:$PATH
LC_ALL=C.UTF-8
LANG=C.UTF-8

# Activate common conda env
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Run isort and get results/return code
ISORT=`isort --check-only python --settings-path=python/setup.cfg `
ISORT_RETVAL=$?

# Run black and get results/return code
BLACK=`black --config python/pyproject.toml --check python`
BLACK_RETVAL=$?

# Run flake8 and get results/return code
FLAKE=`flake8 --config=python/.flake8 python`
FLAKE_RETVAL=$?

# Run flake8-cython and get results/return code
FLAKE_CYTHON=`flake8 --config=python/.flake8.cython`
FLAKE_CYTHON_RETVAL=$?

# Run clang-format and check for a consistent code format
CLANG_FORMAT=`python scripts/run-clang-format.py 2>&1`
CLANG_FORMAT_RETVAL=$?

# Run cmake-format / cmake-lint and get results/return code
CMAKE_FILES=(`find | grep -E "^.*\.cmake(\.in)?$|^.*/CMakeLists.txt$"`)

CMAKE_FORMATS=()
CMAKE_FORMAT_RETVAL=0

CMAKE_LINTS=()
CMAKE_LINT_RETVAL=0

CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[2]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}
gpuci_retry curl -s https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${CURRENT_SHORT_TAG}/cmake-format-rapids-cmake.json -o cmake/rapids-cmake.json

for cmake_file in "${CMAKE_FILES[@]}"; do
  cmake-format --in-place --config-files cmake/config.json cmake/rapids-cmake.json  -- ${cmake_file}
  TMP_CMAKE_FORMAT=`git diff --color --exit-code -- ${cmake_file}`
  TMP_CMAKE_FORMAT_RETVAL=$?
  if [ "$TMP_CMAKE_FORMAT_RETVAL" != "0" ]; then
    CMAKE_FORMAT_RETVAL=1
    CMAKE_FORMATS+=("$TMP_CMAKE_FORMAT")
  fi

  TMP_CMAKE_LINT=`cmake-lint --config-files cmake/config.json cmake/rapids-cmake.json  -- ${cmake_file}`
  TMP_CMAKE_LINT_RETVAL=$?
  if [ "$TMP_CMAKE_LINT_RETVAL" != "0" ]; then
    CMAKE_LINT_RETVAL=1
    CMAKE_LINTS+=("$TMP_CMAKE_LINT")
  fi
done


# Output results if failure otherwise show pass
if [ "$ISORT_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: isort style check; begin output\n\n"
  echo -e "$ISORT"
  echo -e "\n\n>>>> FAILED: isort style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: isort style check\n\n"
fi

if [ "$BLACK_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: black style check; begin output\n\n"
  echo -e "$BLACK"
  echo -e "\n\n>>>> FAILED: black style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: black style check\n\n"
fi

if [ "$FLAKE_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: flake8 style check; begin output\n\n"
  echo -e "$FLAKE"
  echo -e "\n\n>>>> FAILED: flake8 style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: flake8 style check\n\n"
fi

if [ "$FLAKE_CYTHON_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: flake8-cython style check; begin output\n\n"
  echo -e "$FLAKE_CYTHON"
  echo -e "\n\n>>>> FAILED: flake8-cython style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: flake8-cython style check\n\n"
fi

if [ "$CLANG_FORMAT_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: clang format check; begin output\n\n"
  echo -e "$CLANG_FORMAT"
  echo -e "\n\n>>>> FAILED: clang format check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: clang format check\n\n"
fi

if [ "$CMAKE_FORMAT_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: cmake format check; begin output\n\n"
  for CMAKE_FORMAT in "${CMAKE_FORMATS[@]}"; do
    echo -e "$CMAKE_FORMAT"
    echo -e "\n"
  done
  echo -e "\n\n>>>> FAILED: cmake format check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: cmake format check\n\n"
fi

if [ "$CMAKE_LINT_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: cmake lint check; begin output\n\n"
  for CMAKE_LINT in "${CMAKE_LINTS[@]}"; do
    echo -e "$CMAKE_LINT"
    echo -e "\n"
  done
  echo -e "\n\n>>>> FAILED: cmake lint check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: cmake lint check\n\n"
fi

RETVALS=($ISORT_RETVAL $BLACK_RETVAL $FLAKE_RETVAL $FLAKE_CYTHON_RETVAL $CLANG_FORMAT_RETVAL $CMAKE_FORMAT_RETVAL $CMAKE_LINT_RETVAL)
IFS=$'\n'
RETVAL=`echo "${RETVALS[*]}" | sort -nr | head -n1`

exit $RETVAL
