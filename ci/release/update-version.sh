#!/bin/bash
########################
# RMM Version Updater #
########################

## Usage
# bash update-version.sh <new_version>


# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[2]}')
CURRENT_PATCH=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[3]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# cpp update
sed_runner 's/'"  VERSION .*"'/'"  VERSION ${NEXT_FULL_TAG}"'/g' CMakeLists.txt
sed_runner 's/'"branch-.*\/RAPIDS.cmake"'/'"branch-${NEXT_SHORT_TAG}\/RAPIDS.cmake"'/g' CMakeLists.txt

# Python update
sed_runner 's/'"rmm_version .*)"'/'"rmm_version ${NEXT_FULL_TAG})"'/g' python/CMakeLists.txt
sed_runner 's/'"branch-.*\/RAPIDS.cmake"'/'"branch-${NEXT_SHORT_TAG}\/RAPIDS.cmake"'/g' python/CMakeLists.txt
sed_runner "s/__version__ = .*/__version__ = \"${NEXT_FULL_TAG}\"/g" python/rmm/__init__.py
sed_runner "s/^version = .*/version = \"${NEXT_FULL_TAG}\"/g" python/pyproject.toml

# cmake-format rapids-cmake definitions
sed_runner 's/'"branch-.*\/cmake-format-rapids-cmake.json"'/'"branch-${NEXT_SHORT_TAG}\/cmake-format-rapids-cmake.json"'/g' ci/check_style.sh

# doxyfile update
sed_runner 's/'"PROJECT_NUMBER         = .*"'/'"PROJECT_NUMBER         = ${NEXT_SHORT_TAG}"'/g' doxygen/Doxyfile

# sphinx docs update
sed_runner 's/'"version =.*"'/'"version = \"${NEXT_SHORT_TAG}\""'/g' python/docs/conf.py
sed_runner 's/'"release =.*"'/'"release = \"${NEXT_FULL_TAG}\""'/g' python/docs/conf.py

# CI files
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-action-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
done
sed_runner "s/RAPIDS_VERSION_NUMBER=\".*/RAPIDS_VERSION_NUMBER=\"${NEXT_SHORT_TAG}\"/g" ci/build_docs.sh
