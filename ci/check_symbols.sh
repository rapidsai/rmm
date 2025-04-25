#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -eEuo pipefail

echo "checking for symbol visibility issues"

WHEEL_FILE=${1}

raise-symbols-found-error() {
    local pattern="${1}"

err_msg="ERROR: Found some exported symbols matching the pattern '${pattern}'.

These should be marked with 'hidden' visibility.
See https://cmake.org/cmake/help/latest/prop_tgt/LANG_VISIBILITY_PRESET.html and https://gcc.gnu.org/wiki/Visibility for details.
"

    echo ""
    echo "${err_msg}"
    exit 1
}

WHEEL_EXPORT_DIR="$(mktemp -d)"

unzip \
    -d "${WHEEL_EXPORT_DIR}" \
    "${WHEEL_FILE}"

dso_files=$(
    find \
        "${WHEEL_EXPORT_DIR}" \
        -type f \
        \( -name '*.so' -o -name '*.so.*' \)
)

for dso_file in ${dso_files}; do
    echo ""
    echo "checking exported symbols in '${dso_file}'"
    symbol_file="./syms.txt"
    readelf --symbols --wide "${dso_file}" \
        | c++filt \
        > "${symbol_file}"

    echo "symbol counts by type"
    echo "  * GLOBAL: $(grep --count -E ' GLOBAL ' < ${symbol_file})"
    echo "  * WEAK: $(grep --count -E '  WEAK '    < ${symbol_file})"
    echo "  * LOCAL: $(grep --count -E '  LOCAL '  < ${symbol_file})"

    echo "No symbol visibility issues found"
done

echo ""
echo "No symbol visibility issues found in any DSOs"
