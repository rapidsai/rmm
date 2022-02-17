# This code is adapted directly from the scikit-build project under the following license.

# The MIT License (MIT)
# 
# Copyright (c) 2014 Mike Sarahan
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# 
# This project borrows a great deal from the setup tools of the PyNE project.  Here is its license:
# 
# Copyright 2011-2014, the PyNE Development Team. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
# 
#    1. Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
# 
#    2. Redistributions in binary form must reproduce the above copyright notice, this list
#       of conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE PYNE DEVELOPMENT TEAM ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of the stakeholders of the PyNE project or the employers of PyNE developers.

# TODO: Should we guard this based on a scikit-build version? Override this function to avoid
# scikit-build clobbering symbol visibility.
function(_set_python_extension_symbol_visibility _target)
  if(PYTHON_VERSION_MAJOR VERSION_GREATER 2)
    set(_modinit_prefix "PyInit_")
  else()
    set(_modinit_prefix "init")
  endif()
  if("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    set_target_properties(${_target} PROPERTIES LINK_FLAGS "/EXPORT:${_modinit_prefix}${_target}")
  elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(_script_path ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_target}-version-script.map)
    file(
      WRITE ${_script_path}
      # Note: The change is to this script, which does not indiscriminately mark all non PyInit
      # symbols as local.
      "{global: ${_modinit_prefix}${_target}; };")
    set_property(
      TARGET ${_target}
      APPEND_STRING
      PROPERTY LINK_FLAGS " -Wl,--version-script=\"${_script_path}\"")
  endif()
endfunction()
