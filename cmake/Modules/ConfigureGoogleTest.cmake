set(GTEST_ROOT "${CMAKE_CURRENT_BINARY_DIR}/googletest")

set(GTEST_CMAKE_ARGS "")
		     # " -Dgtest_build_samples=ON" 
                     # " -DCMAKE_VERBOSE_MAKEFILE=ON")

# Workaround for https://github.com/google/googletest/issues/854
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(APPEND GTEST_CMAKE_ARGS " -DCMAKE_C_FLAGS=-fPIC")
  list(APPEND GTEST_CMAKE_ARGS " -DCMAKE_CXX_FLAGS=-fPIC")
endif(CMAKE_CXX_COMPILER MATCHES "Clang")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Templates/GoogleTest.CMakeLists.txt.cmake"
               "${GTEST_ROOT}/CMakeLists.txt")

file(MAKE_DIRECTORY "${GTEST_ROOT}/build")
file(MAKE_DIRECTORY "${GTEST_ROOT}/install")

execute_process(COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} .
                RESULT_VARIABLE GTEST_CONFIG
                WORKING_DIRECTORY ${GTEST_ROOT})

if(GTEST_CONFIG)
    message(FATAL_ERROR "Configuring GoogleTest failed: " ${GTEST_CONFIG})
endif(GTEST_CONFIG)

set(PARALLEL_BUILD -j)
if($ENV{PARALLEL_LEVEL})
    set(NUM_JOBS $ENV{PARALLEL_LEVEL})
    set(PARALLEL_BUILD "${PARALLEL_BUILD}${NUM_JOBS}")
endif($ENV{PARALLEL_LEVEL})

if(${NUM_JOBS})
    if(${NUM_JOBS} EQUAL 1)
        message(STATUS "GTEST BUILD: Enabling Sequential CMake build")
    elseif(${NUM_JOBS} GREATER 1)
        message(STATUS "GTEST BUILD: Enabling Parallel CMake build with ${NUM_JOBS} jobs")
    endif(${NUM_JOBS} EQUAL 1)
else()
    message(STATUS "GTEST BUILD: Enabling Parallel CMake build with all threads")
endif(${NUM_JOBS})

execute_process(COMMAND ${CMAKE_COMMAND} --build .. -- ${PARALLEL_BUILD}
                RESULT_VARIABLE GTEST_BUILD
                WORKING_DIRECTORY ${GTEST_ROOT}/build)

if(GTEST_BUILD)
    message(FATAL_ERROR "Building GoogleTest failed: " ${GTEST_BUILD})
endif(GTEST_BUILD)

message(STATUS "GoogleTest installed here: " ${GTEST_ROOT}/install)
set(GTEST_INCLUDE_DIR "${GTEST_ROOT}/install/include")
set(GTEST_LIBRARY_DIR "${GTEST_ROOT}/install/lib")
set(GTEST_FOUND TRUE)

foreach(_lib gtest gtest_main gmock gmock_main)
  add_library(${_lib} STATIC IMPORTED)
  set_target_properties(${_lib} PROPERTIES
    IMPORTED_LOCATION "${GTEST_LIBRARY_DIR}/lib${_lib}.a"
    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
    )
endforeach()
