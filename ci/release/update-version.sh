#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
########################
# RMM Version Updater #
########################

## Usage
# Primary interface:   bash update-version.sh <new_version> [--run-context=update_dev_branch|update_release_branch]
# Fallback interface:  [RUN_CONTEXT=update_dev_branch|update_release_branch] bash update-version.sh <new_version>
# CLI arguments take precedence over environment variables
# Defaults to update_dev_branch (main branch) when no run-context is specified


# Parse command line arguments
CLI_RUN_CONTEXT=""
VERSION_ARG=""

for arg in "$@"; do
    case $arg in
        --run-context=*)
            CLI_RUN_CONTEXT="${arg#*=}"
            shift
            ;;
        *)
            if [[ -z "$VERSION_ARG" ]]; then
                VERSION_ARG="$arg"
            fi
            ;;
    esac
done

# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG="$VERSION_ARG"

# Determine RUN_CONTEXT with CLI precedence over environment variable, defaulting to dev branch
if [[ -n "$CLI_RUN_CONTEXT" ]]; then
    RUN_CONTEXT="$CLI_RUN_CONTEXT"
    echo "Using RUN_CONTEXT from CLI: $RUN_CONTEXT"
elif [[ -n "${RUN_CONTEXT}" ]]; then
    echo "Using RUN_CONTEXT from environment: $RUN_CONTEXT"
else
    RUN_CONTEXT="update_dev_branch"
    echo "No RUN_CONTEXT provided, defaulting to: $RUN_CONTEXT (main branch)"
fi

# Validate RUN_CONTEXT value
if [[ "${RUN_CONTEXT}" != "update_dev_branch" && "${RUN_CONTEXT}" != "update_release_branch" ]]; then
    echo "Error: Invalid RUN_CONTEXT value '${RUN_CONTEXT}'"
    echo "Valid values: update_dev_branch, update_release_branch"
    exit 1
fi

# Validate version argument
if [[ -z "$NEXT_FULL_TAG" ]]; then
    echo "Error: Version argument is required"
    echo "Usage: $0 <new_version> [--run-context=<context>]"
    echo "   or: [RUN_CONTEXT=<context>] $0 <new_version>"
    echo "Note: Defaults to update_dev_branch (main branch) when run-context is not specified"
    exit 1
fi

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')

# Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

# Need to distutils-normalize the original version
NEXT_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_SHORT_TAG}'))")

# Set branch references based on RUN_CONTEXT
if [[ "${RUN_CONTEXT}" == "update_dev_branch" ]]; then
    RAPIDS_BRANCH_NAME="main"
    WORKFLOW_BRANCH_REF="main"
    echo "Preparing development branch update $CURRENT_TAG => $NEXT_FULL_TAG (targeting main branch)"
elif [[ "${RUN_CONTEXT}" == "update_release_branch" ]]; then
    RAPIDS_BRANCH_NAME="release/${NEXT_SHORT_TAG}"
    WORKFLOW_BRANCH_REF="release/${NEXT_SHORT_TAG}"
    echo "Preparing release branch update $CURRENT_TAG => $NEXT_FULL_TAG (targeting release/${NEXT_SHORT_TAG} branch)"
fi

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}

# Centralized version file update
echo "${NEXT_FULL_TAG}" > VERSION
echo "${RAPIDS_BRANCH_NAME}" > RAPIDS_BRANCH

# Examples update
sed_runner "s/RMM_TAG release\/[0-9.]*/RMM_TAG ${RAPIDS_BRANCH_NAME}/" cpp/examples/versions.cmake
sed_runner "s/RMM_TAG main/RMM_TAG ${RAPIDS_BRANCH_NAME}/" cpp/examples/versions.cmake

# CI files
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@${WORKFLOW_BRANCH_REF}/g" "${FILE}"
  sed_runner "s/:[0-9]*\\.[0-9]*-/:${NEXT_SHORT_TAG}-/g" "${FILE}"
done

# .devcontainer files
find .devcontainer/ -type f -name devcontainer.json -print0 | while IFS= read -r -d '' filename; do
    sed_runner "s@rapidsai/devcontainers:[0-9.]*@rapidsai/devcontainers:${NEXT_SHORT_TAG}@g" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/rapids-build-utils:[0-9.]*@rapidsai/devcontainers/features/rapids-build-utils:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapids-\${localWorkspaceFolderBasename}-[0-9.]*@rapids-\${localWorkspaceFolderBasename}-${NEXT_SHORT_TAG}@g" "${filename}"
done

DEPENDENCIES=(
  librmm
  librmm-tests
  librmm-example
  rmm
)
for DEP in "${DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml conda/environments/*.yaml; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
  for FILE in python/*/pyproject.toml; do
    sed_runner "/\"${DEP}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0\"/g" "${FILE}"
  done
done
