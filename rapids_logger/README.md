# About

The `rapids-logger` project defines an easy way to produce a project-specific logger using the excellent [spdlog](https://github.com/gabime/spdlog) package.
The goal of this project is to ensure that projects wishing to provide their own logger may do so easily without needing to reimplement their own custom wrappers around spdlog.
A core goal of the project is to ensure that the custom logger implementation does not leak any spdlog symbols, allowing the safe coexistence of different projects in the same environment even if they use different versions of spdlog.
That goal is the primary reason to prefer using this project rather than directly exposing a specialized instance of a spdlog logger in your own project.

`rapids-logger` is designed to be used via CMake.
Its CMake defines a function `rapids_make_logger` that can be used to produce a project-specific logger class in a provided namespace.
The resulting logger exposes spdlog-like functionality via the [PImpl idiom](https://en.cppreference.com/w/cpp/language/pimpl) to avoid exposing spdlog symbols publicly.
It uses CMake and template C++ files to generate a public header file to describe the user interface and an inline header that should be placed in a single TU by consumers to compile the implementation.
To simplify usage, each invocation of the function produces two CMake targets, one representing the public header and one representing a trivial source file including the inline header.
Projects using `rapids-logger` should make the first target part of their public link interface while the latter should be linked to privately so that it is compiled into the project's library without public exposure.

To support transitive usage in header-only libraries that may not wish to enforce the spdlog requirement on their consumers, `rapids-logger` may be compiled with `SUPPORTS_LOGGING=OFF`, in which case a nonfunctional logger is produced instead.
In this case the inline header is never used, and spdlog is not required at build-time.
All logging functions and macros are defined, so any code written using the logger will still compile, but no actual logging will take place (and any code that assumes logs have been written will fail at runtime).

To mirror spdlog, each generated logger also ships with a set of logging macros `<project-name>_LOG_<log-level>` that may be used to control logging at compile-time as well as runtime using a compile-time variable `<project-name>_LOG_ACTIVE_LEVEL`.
For example, a project called "rapids" will be able to write code like this:
```
RAPIDS_LOG_WARN("Some message to be shown when the warning level is enabled");
```
and control whether that warning is shown by compiling the code with `RAPIDS_LOG_ACTIVE_LEVEL=RAPIDS_LOG_LEVEL_WARN`.
Each project is endowed with its own definition of levels, so different projects in the same environment may be safely configured independently of each other and of spdlog.
Each project is also given a `default_logger` function that produces a global logger that may be used anywhere, but projects may also freely instantiate additional loggers as needed.
