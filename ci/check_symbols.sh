#!/bin/bash

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

    # Explanation for '-v' uses here:
    #
    #   * 'format_error' symbols are intentionally exported, that type of error
    #      can be thrown across library boundaries. See "Problems with C++ exceptions"
    #      at https://gcc.gnu.org/wiki/Visibility.
    echo "checking for 'fmt::' symbols..."
    if grep -E 'fmt\:\:' < "${symbol_file}" \
        | grep -v 'format_error'
    then
        raise-symbols-found-error 'fmt::'
    fi

    # Explanation for '-v' uses here:
    #
    #  * trivially-destructible objects sometimes get an entry in the symbol table
    #    for a specialization of `std::_Destroy_aux()` called to destroy them.
    #    There is one for `spdlog::details::log_msg_buffer like that:
    #
    #       'std::_Destroy_aux<false>::__destroy<spdlog::details::log_msg_buffer*>'
    #
    #    That should be safe to export.
    #
    echo "checking for 'spdlog::' symbols..."
    if grep -E 'spdlog\:\:' < "${symbol_file}" \
        | grep -v 'std\:\:_Destroy_aux'
    then
        raise-symbols-found-error 'spdlog::'
    fi
    echo "No symbol visibility issues found"
done

echo ""
echo "No symbol visibility issues found in any DSOs"
