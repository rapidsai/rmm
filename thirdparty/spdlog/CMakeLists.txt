include(FetchContent)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.6.0
    GIT_SHALLOW true
)

FetchContent_MakeAvailable(spdlog)
