#[=======================================================================[.rst:
add_cython_modules
------------------

Generate C(++) from Cython and create Python modules.

.. code-block:: cmake

  add_cython_modules(<ModuleName...>)

Creates a Cython target for a module, then adds a corresponding Python
extension module.

``ModuleName``
  The list of modules to build.

#]=======================================================================]
function(add_cython_modules cython_modules)
  foreach(cython_module ${cython_modules})
    add_cython_target(${cython_module} CXX PY3)
    add_library(${cython_module} MODULE ${cython_module})
    # TODO: This fails unless the C language is enabled in the project.Also,
    # this is slower to compile than just linking manually to Python3 below.
    python_extension_module(${cython_module})

    # To avoid libraries being prefixed with "lib".
    set_target_properties(${cython_module} PROPERTIES PREFIX "")

    # TODO: This is the "manual" version of the python_extension_module command
    # above, but it seems to compile faster.
    #target_link_libraries(${cython_module} rmm::rmm Python3::Module)
    target_link_libraries(${cython_module} rmm::rmm)
    target_include_directories(${cython_module} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    install(TARGETS ${cython_module} DESTINATION rmm/_lib)
  endforeach(cython_module ${cython_sources})
endfunction()
