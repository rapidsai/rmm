function (write_version)
  message(STATUS "RMM VERSION: ${RMM_VERSION}")
  configure_file(
    ${RMM_SOURCE_DIR}/cmake/version_config.hpp.in
    ${RMM_BINARY_DIR}/include/rmm/version_config.hpp @ONLY)
endfunction (write_version)
