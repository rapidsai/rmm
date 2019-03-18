#=============================================================================
# Copyright (c) 2018, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(RMM VERSION 0.6.0 LANGUAGES C CXX CUDA)

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

    option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)
    if(CMAKE_CXX11_ABI)
        message(STATUS "CUDF: Enabling the GLIBCXX11 ABI")
    else()
        message(STATUS "CUDF: Disabling the GLIBCXX11 ABI")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    endif(CMAKE_CXX11_ABI)
endif(CMAKE_COMPILER_IS_GNUCXX)

option(BUILD_TESTS "Configure CMake to build tests"
       ON)

###################################################################################################
# - cnmem ---------------------------------------------------------------------------------
# include cnmem.h in the include/detail directory
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cnmem/include/cnmem.h" 
               "${CMAKE_CURRENT_SOURCE_DIR}/include/rmm/detail/cnmem.h" COPYONLY)


###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)

###################################################################################################
# - add gtest -------------------------------------------------------------------------------------

if(BUILD_TESTS)
    include(CTest)
    include(ConfigureGoogleTest)

    if(GTEST_FOUND)
        message(STATUS "Google C++ Testing Framework (Google Test) found in ${GTEST_ROOT}")
        include_directories(${GTEST_INCLUDE_DIR})
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
    else()
        message(AUTHOR_WARNING "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
    endif(GTEST_FOUND)
endif(BUILD_TESTS)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src"
                    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cnmem/include")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_CURRENT_BINARY_DIR}/lib"
                 "${GTEST_LIBRARY_DIR}")

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(rmm SHARED
            src/rmm.cpp
            src/memory_manager.cpp
            thirdparty/cnmem/src/cnmem.cpp)

# override rpath for rmm
SET_TARGET_PROPERTIES(rmm PROPERTIES BUILD_RPATH "\$ORIGIN")

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
if(USE_NVTX)
    message(STATUS "Using Nvidia Tools Extension")
    find_library(NVTX_LIBRARY nvToolsExt PATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    target_link_libraries(rmm ${NVTX_LIBRARY})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --define-macro USE_NVTX")
endif(USE_NVTX)

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(rmm cudart cuda)

###################################################################################################
# - python cffi bindings --------------------------------------------------------------------------

# To enable building RMM as a submodule, where the binary directory is not underneath the RMM
# source directory.
set(RMM_API_H_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include/rmm/rmm_api.h")
# Input is a template which assigns MEMORY_H_PATH to a variable. This command replaces the variable with its value
# and saves the file as librmm_build.py
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/librmm_cffi/librmm_build.py.in" 
               "${CMAKE_CURRENT_SOURCE_DIR}/python/librmm_cffi/librmm_build.py")

add_custom_command(OUTPUT RMM_PYTHON_CFFI
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                   COMMAND test ! -e ${CMAKE_BINARY_DIR}/rmm/librmm.so || ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/rmm/librmm.so ${CMAKE_BINARY_DIR}/librmm.so
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/librmm_cffi ${CMAKE_BINARY_DIR}/python/librmm_cffi
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/tests ${CMAKE_BINARY_DIR}/python/tests
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py ${CMAKE_BINARY_DIR}/python/rmm_setup.py
                   COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR}/python python rmm_setup.py build_ext --inplace
                   VERBATIM)

add_custom_target(rmm_python_cffi DEPENDS rmm RMM_PYTHON_CFFI)

###################################################################################################
# - install targets -------------------------------------------------------------------------------

install(TARGETS rmm
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/rmm
        DESTINATION include)

add_custom_command(OUTPUT RMM_INSTALL_PYTHON_CFFI
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                   COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR}/python python rmm_setup.py install --single-version-externally-managed --record=record.txt
                   VERBATIM)

add_custom_target(rmm_install_python DEPENDS rmm RMM_PYTHON_CFFI RMM_INSTALL_PYTHON_CFFI)

###################################################################################################
# - make documentation ----------------------------------------------------------------------------

add_custom_command(OUTPUT RMM_DOXYGEN
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
                   COMMAND doxygen Doxyfile
                   VERBATIM)

add_custom_target(rmm_doc DEPENDS RMM_DOXYGEN)
