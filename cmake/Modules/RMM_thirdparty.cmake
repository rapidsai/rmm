# spdlog

set(RMM_MIN_VERSION_spdlog 1.7.0)

CPMFindPackage(
  NAME spdlog
  GITHUB_REPOSITORY gabime/spdlog
  VERSION ${RMM_MIN_VERSION_spdlog}
  GIT_SHALLOW TRUE
  # If there is no pre-installed spdlog we can use, we'll install our fetched copy together with RMM
  OPTIONS "SPDLOG_INSTALL TRUE")

# thrust/cub

set(RMM_MIN_VERSION_Thrust 1.9.0)

CPMFindPackage(
  NAME Thrust
  GITHUB_REPOSITORY NVIDIA/thrust
  GIT_TAG 1.10.0
  VERSION 1.10.0
  GIT_SHALLOW TRUE
  # If there is no pre-installed thrust we can use, we'll install our fetched copy together with RMM
  OPTIONS "THRUST_INSTALL TRUE")

thrust_create_target(rmm::Thrust FROM_OPTIONS)
