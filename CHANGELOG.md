# RMM 23.06.00 (7 Jun 2023)

## 🚨 Breaking Changes

- Update minimum Python version to Python 3.9 ([#1252](https://github.com/rapidsai/rmm/pull/1252)) [@shwina](https://github.com/shwina)

## 🐛 Bug Fixes

- Ensure Logger tests aren&#39;t run in parallel ([#1277](https://github.com/rapidsai/rmm/pull/1277)) [@robertmaynard](https://github.com/robertmaynard)
- Pin to scikit-build&lt;0.17.2. ([#1262](https://github.com/rapidsai/rmm/pull/1262)) [@bdice](https://github.com/bdice)

## 🛠️ Improvements

- Require Numba 0.57.0+ &amp; NumPy 1.21.0+ ([#1279](https://github.com/rapidsai/rmm/pull/1279)) [@jakirkham](https://github.com/jakirkham)
- Align test_cpp.sh with conventions in other RAPIDS repos. ([#1269](https://github.com/rapidsai/rmm/pull/1269)) [@bdice](https://github.com/bdice)
- Switch back to using primary shared-action-workflows branch ([#1268](https://github.com/rapidsai/rmm/pull/1268)) [@vyasr](https://github.com/vyasr)
- Update recipes to GTest version &gt;=1.13.0 ([#1263](https://github.com/rapidsai/rmm/pull/1263)) [@bdice](https://github.com/bdice)
- Support CUDA 12.0 for pip wheels ([#1259](https://github.com/rapidsai/rmm/pull/1259)) [@bdice](https://github.com/bdice)
- Add build vars ([#1258](https://github.com/rapidsai/rmm/pull/1258)) [@AyodeAwe](https://github.com/AyodeAwe)
- Enable sccache hits from local builds ([#1257](https://github.com/rapidsai/rmm/pull/1257)) [@AyodeAwe](https://github.com/AyodeAwe)
- Revert to branch-23.06 for shared-action-workflows ([#1256](https://github.com/rapidsai/rmm/pull/1256)) [@shwina](https://github.com/shwina)
- run docs builds nightly too ([#1255](https://github.com/rapidsai/rmm/pull/1255)) [@AyodeAwe](https://github.com/AyodeAwe)
- Build wheels using new single image workflow ([#1254](https://github.com/rapidsai/rmm/pull/1254)) [@vyasr](https://github.com/vyasr)
- Update minimum Python version to Python 3.9 ([#1252](https://github.com/rapidsai/rmm/pull/1252)) [@shwina](https://github.com/shwina)
- Remove usage of rapids-get-rapids-version-from-git ([#1251](https://github.com/rapidsai/rmm/pull/1251)) [@jjacobelli](https://github.com/jjacobelli)
- Remove wheel pytest verbosity ([#1249](https://github.com/rapidsai/rmm/pull/1249)) [@sevagh](https://github.com/sevagh)
- Update clang-format to 16.0.1. ([#1246](https://github.com/rapidsai/rmm/pull/1246)) [@bdice](https://github.com/bdice)
- Remove uses-setup-env-vars ([#1242](https://github.com/rapidsai/rmm/pull/1242)) [@vyasr](https://github.com/vyasr)
- Move RMM_LOGGING_ASSERT into separate header ([#1241](https://github.com/rapidsai/rmm/pull/1241)) [@ahendriksen](https://github.com/ahendriksen)
- Use ARC V2 self-hosted runners for GPU jobs ([#1239](https://github.com/rapidsai/rmm/pull/1239)) [@jjacobelli](https://github.com/jjacobelli)

# RMM 23.04.00 (6 Apr 2023)

## 🐛 Bug Fixes

- Remove MANIFEST.in use auto-generated one for sdists and package_data for wheels ([#1233](https://github.com/rapidsai/rmm/pull/1233)) [@vyasr](https://github.com/vyasr)
- Fix update-version.sh. ([#1227](https://github.com/rapidsai/rmm/pull/1227)) [@vyasr](https://github.com/vyasr)
- Specify include_package_data to setup ([#1218](https://github.com/rapidsai/rmm/pull/1218)) [@vyasr](https://github.com/vyasr)
- Revert changes overriding rapids-cmake repo. ([#1209](https://github.com/rapidsai/rmm/pull/1209)) [@bdice](https://github.com/bdice)
- Synchronize stream in `DeviceBuffer.c_from_unique_ptr` constructor ([#1100](https://github.com/rapidsai/rmm/pull/1100)) [@shwina](https://github.com/shwina)

## 🚀 New Features

- Use rapids-cmake parallel testing feature ([#1183](https://github.com/rapidsai/rmm/pull/1183)) [@robertmaynard](https://github.com/robertmaynard)

## 🛠️ Improvements

- Stop setting package version attribute in wheels ([#1236](https://github.com/rapidsai/rmm/pull/1236)) [@vyasr](https://github.com/vyasr)
- Add codespell as a linter ([#1231](https://github.com/rapidsai/rmm/pull/1231)) [@bdice](https://github.com/bdice)
- Pass `AWS_SESSION_TOKEN` and `SCCACHE_S3_USE_SSL` vars to conda build ([#1230](https://github.com/rapidsai/rmm/pull/1230)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update to GCC 11 ([#1228](https://github.com/rapidsai/rmm/pull/1228)) [@bdice](https://github.com/bdice)
- Fix some minor oversights in the conversion to pyproject.toml ([#1226](https://github.com/rapidsai/rmm/pull/1226)) [@vyasr](https://github.com/vyasr)
- Remove pickle compatibility layer in tests for Python &lt; 3.8. ([#1224](https://github.com/rapidsai/rmm/pull/1224)) [@bdice](https://github.com/bdice)
- Move external allocators into rmm.allocators module to defer imports ([#1221](https://github.com/rapidsai/rmm/pull/1221)) [@wence-](https://github.com/wence-)
- Generate pyproject.toml dependencies using dfg ([#1219](https://github.com/rapidsai/rmm/pull/1219)) [@vyasr](https://github.com/vyasr)
- Run rapids-dependency-file-generator via pre-commit ([#1217](https://github.com/rapidsai/rmm/pull/1217)) [@vyasr](https://github.com/vyasr)
- Skip docs job in nightly runs ([#1215](https://github.com/rapidsai/rmm/pull/1215)) [@AyodeAwe](https://github.com/AyodeAwe)
- CI: Remove specification of manual stage for check_style.sh script. ([#1214](https://github.com/rapidsai/rmm/pull/1214)) [@csadorf](https://github.com/csadorf)
- Use script rather than environment variable to modify package names ([#1212](https://github.com/rapidsai/rmm/pull/1212)) [@vyasr](https://github.com/vyasr)
- Reduce error handling verbosity in CI tests scripts ([#1204](https://github.com/rapidsai/rmm/pull/1204)) [@AjayThorve](https://github.com/AjayThorve)
- Update shared workflow branches ([#1203](https://github.com/rapidsai/rmm/pull/1203)) [@ajschmidt8](https://github.com/ajschmidt8)
- Use date in build string instead of in the version. ([#1195](https://github.com/rapidsai/rmm/pull/1195)) [@bdice](https://github.com/bdice)
- Stop using versioneer to manage versions ([#1190](https://github.com/rapidsai/rmm/pull/1190)) [@vyasr](https://github.com/vyasr)
- Update to spdlog&gt;=1.11.0, fmt&gt;=9.1.0. ([#1177](https://github.com/rapidsai/rmm/pull/1177)) [@bdice](https://github.com/bdice)
- Migrate as much as possible to `pyproject.toml` ([#1151](https://github.com/rapidsai/rmm/pull/1151)) [@jakirkham](https://github.com/jakirkham)

# RMM 23.02.00 (9 Feb 2023)

## 🐛 Bug Fixes

- pre-commit: Update isort version to 5.12.0 ([#1197](https://github.com/rapidsai/rmm/pull/1197)) [@wence-](https://github.com/wence-)
- Revert &quot;Upgrade to spdlog 1.10 ([#1173)&quot; (#1176](https://github.com/rapidsai/rmm/pull/1173)&quot; (#1176)) [@bdice](https://github.com/bdice)
- Ensure `UpstreamResourceAdaptor` is not cleared by the Python GC ([#1170](https://github.com/rapidsai/rmm/pull/1170)) [@shwina](https://github.com/shwina)

## 📖 Documentation

- Fix documentation author ([#1188](https://github.com/rapidsai/rmm/pull/1188)) [@bdice](https://github.com/bdice)

## 🚀 New Features

- Add RMM PyTorch allocator ([#1168](https://github.com/rapidsai/rmm/pull/1168)) [@shwina](https://github.com/shwina)

## 🛠️ Improvements

- Update shared workflow branches ([#1201](https://github.com/rapidsai/rmm/pull/1201)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fix update-version.sh ([#1199](https://github.com/rapidsai/rmm/pull/1199)) [@raydouglass](https://github.com/raydouglass)
- Use CTK 118/cp310 branch of wheel workflows ([#1193](https://github.com/rapidsai/rmm/pull/1193)) [@sevagh](https://github.com/sevagh)
- Update `build.yaml` workflow to reduce verbosity ([#1192](https://github.com/rapidsai/rmm/pull/1192)) [@AyodeAwe](https://github.com/AyodeAwe)
- Fix `build.yaml` workflow ([#1191](https://github.com/rapidsai/rmm/pull/1191)) [@ajschmidt8](https://github.com/ajschmidt8)
- add docs_build step ([#1189](https://github.com/rapidsai/rmm/pull/1189)) [@AyodeAwe](https://github.com/AyodeAwe)
- Upkeep/wheel param cleanup ([#1187](https://github.com/rapidsai/rmm/pull/1187)) [@sevagh](https://github.com/sevagh)
- Update workflows for nightly tests ([#1186](https://github.com/rapidsai/rmm/pull/1186)) [@ajschmidt8](https://github.com/ajschmidt8)
- Build CUDA `11.8` and Python `3.10` Packages ([#1184](https://github.com/rapidsai/rmm/pull/1184)) [@ajschmidt8](https://github.com/ajschmidt8)
- Build wheels alongside conda CI ([#1182](https://github.com/rapidsai/rmm/pull/1182)) [@sevagh](https://github.com/sevagh)
- Update conda recipes. ([#1180](https://github.com/rapidsai/rmm/pull/1180)) [@bdice](https://github.com/bdice)
- Update PR Workflow ([#1174](https://github.com/rapidsai/rmm/pull/1174)) [@ajschmidt8](https://github.com/ajschmidt8)
- Upgrade to spdlog 1.10 ([#1173](https://github.com/rapidsai/rmm/pull/1173)) [@kkraus14](https://github.com/kkraus14)
- Enable `codecov` ([#1171](https://github.com/rapidsai/rmm/pull/1171)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add support for Python 3.10. ([#1166](https://github.com/rapidsai/rmm/pull/1166)) [@bdice](https://github.com/bdice)
- Update pre-commit hooks ([#1154](https://github.com/rapidsai/rmm/pull/1154)) [@bdice](https://github.com/bdice)

# RMM 22.12.00 (8 Dec 2022)

## 🐛 Bug Fixes

- Don&#39;t use CMake 3.25.0 as it has a show stopping FindCUDAToolkit bug ([#1162](https://github.com/rapidsai/rmm/pull/1162)) [@robertmaynard](https://github.com/robertmaynard)
- Relax test for async memory pool IPC handle support ([#1130](https://github.com/rapidsai/rmm/pull/1130)) [@bdice](https://github.com/bdice)

## 📖 Documentation

- Use rapidsai CODE_OF_CONDUCT.md ([#1159](https://github.com/rapidsai/rmm/pull/1159)) [@bdice](https://github.com/bdice)
- Fix doxygen formatting for set_stream. ([#1153](https://github.com/rapidsai/rmm/pull/1153)) [@bdice](https://github.com/bdice)
- Document required Python dependencies to build from source ([#1146](https://github.com/rapidsai/rmm/pull/1146)) [@ccoulombe](https://github.com/ccoulombe)
- fix failed automerge (Branch 22.12 merge 22.10) ([#1131](https://github.com/rapidsai/rmm/pull/1131)) [@harrism](https://github.com/harrism)

## 🚀 New Features

- Add wheel builds to rmm ([#1148](https://github.com/rapidsai/rmm/pull/1148)) [@vyasr](https://github.com/vyasr)

## 🛠️ Improvements

- Align __version__ with wheel version ([#1161](https://github.com/rapidsai/rmm/pull/1161)) [@sevagh](https://github.com/sevagh)
- Add `ninja` &amp; Update CI environment variables ([#1155](https://github.com/rapidsai/rmm/pull/1155)) [@ajschmidt8](https://github.com/ajschmidt8)
- Remove CUDA 11.0 from dependencies.yaml. ([#1152](https://github.com/rapidsai/rmm/pull/1152)) [@bdice](https://github.com/bdice)
- Update dependencies schema. ([#1147](https://github.com/rapidsai/rmm/pull/1147)) [@bdice](https://github.com/bdice)
- Enable sccache for python build ([#1145](https://github.com/rapidsai/rmm/pull/1145)) [@Ethyling](https://github.com/Ethyling)
- Remove Jenkins scripts ([#1143](https://github.com/rapidsai/rmm/pull/1143)) [@ajschmidt8](https://github.com/ajschmidt8)
- Use `ninja` in GitHub Actions ([#1142](https://github.com/rapidsai/rmm/pull/1142)) [@ajschmidt8](https://github.com/ajschmidt8)
- Switch to using rapids-cmake for gbench. ([#1139](https://github.com/rapidsai/rmm/pull/1139)) [@vyasr](https://github.com/vyasr)
- Remove stale labeler ([#1137](https://github.com/rapidsai/rmm/pull/1137)) [@raydouglass](https://github.com/raydouglass)
- Add a public `copy` API to `DeviceBuffer` ([#1128](https://github.com/rapidsai/rmm/pull/1128)) [@galipremsagar](https://github.com/galipremsagar)
- Format gdb script. ([#1127](https://github.com/rapidsai/rmm/pull/1127)) [@bdice](https://github.com/bdice)

# RMM 22.10.00 (12 Oct 2022)

## 🐛 Bug Fixes

- Ensure consistent spdlog dependency target no matter the source ([#1101](https://github.com/rapidsai/rmm/pull/1101)) [@robertmaynard](https://github.com/robertmaynard)
- Remove cuda event deadlocking issues in device mr tests ([#1097](https://github.com/rapidsai/rmm/pull/1097)) [@robertmaynard](https://github.com/robertmaynard)
- Propagate exceptions raised in Python callback functions ([#1096](https://github.com/rapidsai/rmm/pull/1096)) [@madsbk](https://github.com/madsbk)
- Avoid unused parameter warnings in do_get_mem_info ([#1084](https://github.com/rapidsai/rmm/pull/1084)) [@fkallen](https://github.com/fkallen)
- Use rapids-cmake 22.10 best practice for RAPIDS.cmake location ([#1083](https://github.com/rapidsai/rmm/pull/1083)) [@robertmaynard](https://github.com/robertmaynard)

## 📖 Documentation

- Document that minimum required CMake version is now 3.23.1 ([#1098](https://github.com/rapidsai/rmm/pull/1098)) [@robertmaynard](https://github.com/robertmaynard)
- Fix docs for module-level API ([#1091](https://github.com/rapidsai/rmm/pull/1091)) [@bdice](https://github.com/bdice)
- Improve DeviceBuffer docs. ([#1090](https://github.com/rapidsai/rmm/pull/1090)) [@bdice](https://github.com/bdice)
- Branch 22.10 merge 22.08 ([#1089](https://github.com/rapidsai/rmm/pull/1089)) [@harrism](https://github.com/harrism)
- Improve docs formatting and update links. ([#1086](https://github.com/rapidsai/rmm/pull/1086)) [@bdice](https://github.com/bdice)
- Add resources section to README. ([#1085](https://github.com/rapidsai/rmm/pull/1085)) [@bdice](https://github.com/bdice)
- Simplify PR template. ([#1080](https://github.com/rapidsai/rmm/pull/1080)) [@bdice](https://github.com/bdice)

## 🚀 New Features

- Add `gdb` pretty-printers for rmm types ([#1088](https://github.com/rapidsai/rmm/pull/1088)) [@upsj](https://github.com/upsj)
- Support using THRUST_WRAPPED_NAMESPACE ([#1077](https://github.com/rapidsai/rmm/pull/1077)) [@robertmaynard](https://github.com/robertmaynard)

## 🛠️ Improvements

- GH Actions - Enforce `checks` before builds run ([#1125](https://github.com/rapidsai/rmm/pull/1125)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update GH Action Workflows ([#1123](https://github.com/rapidsai/rmm/pull/1123)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add `cudatoolkit` versions to `dependencies.yaml` ([#1119](https://github.com/rapidsai/rmm/pull/1119)) [@ajschmidt8](https://github.com/ajschmidt8)
- Remove `rmm` installation from `librmm` tests` ([#1117](https://github.com/rapidsai/rmm/pull/1117)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add GitHub Actions workflows ([#1104](https://github.com/rapidsai/rmm/pull/1104)) [@Ethyling](https://github.com/Ethyling)
- `build.sh`: accept `--help` ([#1093](https://github.com/rapidsai/rmm/pull/1093)) [@madsbk](https://github.com/madsbk)
- Move clang dependency to conda develop packages. ([#1092](https://github.com/rapidsai/rmm/pull/1092)) [@bdice](https://github.com/bdice)
- Add device_uvector::reserve and device_buffer::reserve ([#1079](https://github.com/rapidsai/rmm/pull/1079)) [@upsj](https://github.com/upsj)
- Bifurcate Dependency Lists ([#1073](https://github.com/rapidsai/rmm/pull/1073)) [@ajschmidt8](https://github.com/ajschmidt8)

# RMM 22.08.00 (17 Aug 2022)

## 🐛 Bug Fixes

- Specify `language` as `&#39;en&#39;` instead of `None` ([#1059](https://github.com/rapidsai/rmm/pull/1059)) [@jakirkham](https://github.com/jakirkham)
- Add a missed `except *` ([#1057](https://github.com/rapidsai/rmm/pull/1057)) [@shwina](https://github.com/shwina)
- Properly handle cudaMemHandleTypeNone and cudaErrorInvalidValue in is_export_handle_type_supported ([#1055](https://github.com/rapidsai/rmm/pull/1055)) [@gerashegalov](https://github.com/gerashegalov)

## 📖 Documentation

- Centralize common css &amp; js code in docs ([#1075](https://github.com/rapidsai/rmm/pull/1075)) [@galipremsagar](https://github.com/galipremsagar)

## 🛠️ Improvements

- Add the ability to register and unregister reinitialization hooks ([#1072](https://github.com/rapidsai/rmm/pull/1072)) [@shwina](https://github.com/shwina)
- Update isort to 5.10.1 ([#1069](https://github.com/rapidsai/rmm/pull/1069)) [@vyasr](https://github.com/vyasr)
- Forward merge 22.06 into 22.08 ([#1067](https://github.com/rapidsai/rmm/pull/1067)) [@vyasr](https://github.com/vyasr)
- Forward merge 22.06 into 22.08 ([#1066](https://github.com/rapidsai/rmm/pull/1066)) [@vyasr](https://github.com/vyasr)
- Pin max version of `cuda-python` to `11.7` ([#1062](https://github.com/rapidsai/rmm/pull/1062)) [@galipremsagar](https://github.com/galipremsagar)
- Change build.sh to find C++ library by default and avoid shadowing CMAKE_ARGS ([#1053](https://github.com/rapidsai/rmm/pull/1053)) [@vyasr](https://github.com/vyasr)

# RMM 22.06.00 (7 Jun 2022)

## 🐛 Bug Fixes

- Clarifies Python requirements and version constraints ([#1037](https://github.com/rapidsai/rmm/pull/1037)) [@jakirkham](https://github.com/jakirkham)
- Use `lib` (not `lib64`) for libraries ([#1024](https://github.com/rapidsai/rmm/pull/1024)) [@jakirkham](https://github.com/jakirkham)
- Properly enable Cython docstrings. ([#1020](https://github.com/rapidsai/rmm/pull/1020)) [@vyasr](https://github.com/vyasr)
- Update `RMMNumbaManager` to handle `NUMBA_CUDA_USE_NVIDIA_BINDING=1` ([#1004](https://github.com/rapidsai/rmm/pull/1004)) [@brandon-b-miller](https://github.com/brandon-b-miller)

## 📖 Documentation

- Clarify using RMM with other Python libraries ([#1034](https://github.com/rapidsai/rmm/pull/1034)) [@jrhemstad](https://github.com/jrhemstad)
- Replace `to_device` with `DeviceBuffer.to_device` ([#1033](https://github.com/rapidsai/rmm/pull/1033)) [@wence-](https://github.com/wence-)
- Documentation Fix: Replace `cudf::logic_error` with `rmm::logic_error` ([#1021](https://github.com/rapidsai/rmm/pull/1021)) [@codereport](https://github.com/codereport)

## 🚀 New Features

- Add rmm::exec_policy_nosync ([#1009](https://github.com/rapidsai/rmm/pull/1009)) [@fkallen](https://github.com/fkallen)
- Callback memory resource ([#980](https://github.com/rapidsai/rmm/pull/980)) [@shwina](https://github.com/shwina)

## 🛠️ Improvements

- Fix conda recipes for conda compilers ([#1043](https://github.com/rapidsai/rmm/pull/1043)) [@Ethyling](https://github.com/Ethyling)
- Use new rapids-cython component of rapids-cmake to simplify builds ([#1031](https://github.com/rapidsai/rmm/pull/1031)) [@vyasr](https://github.com/vyasr)
- Merge branch-22.04 to branch-22.06 ([#1028](https://github.com/rapidsai/rmm/pull/1028)) [@jakirkham](https://github.com/jakirkham)
- Update CMake pinning to just avoid 3.23.0. ([#1023](https://github.com/rapidsai/rmm/pull/1023)) [@vyasr](https://github.com/vyasr)
- Build python using conda in GPU jobs ([#1017](https://github.com/rapidsai/rmm/pull/1017)) [@Ethyling](https://github.com/Ethyling)
- Remove pip requirements file. ([#1015](https://github.com/rapidsai/rmm/pull/1015)) [@bdice](https://github.com/bdice)
- Clean up Thrust includes. ([#1011](https://github.com/rapidsai/rmm/pull/1011)) [@bdice](https://github.com/bdice)
- Update black version ([#1010](https://github.com/rapidsai/rmm/pull/1010)) [@vyasr](https://github.com/vyasr)
- Update cmake-format version for pre-commit and environments. ([#995](https://github.com/rapidsai/rmm/pull/995)) [@vyasr](https://github.com/vyasr)
- Use conda compilers ([#977](https://github.com/rapidsai/rmm/pull/977)) [@Ethyling](https://github.com/Ethyling)
- Build conda packages using mambabuild ([#900](https://github.com/rapidsai/rmm/pull/900)) [@Ethyling](https://github.com/Ethyling)

# RMM 22.04.00 (6 Apr 2022)

## 🐛 Bug Fixes

- Add cuda-python dependency to pyproject.toml ([#994](https://github.com/rapidsai/rmm/pull/994)) [@sevagh](https://github.com/sevagh)
- Disable opportunistic reuse in async mr when cuda driver &lt; 11.5 ([#993](https://github.com/rapidsai/rmm/pull/993)) [@rongou](https://github.com/rongou)
- Use CUDA 11.2+ features via dlopen ([#990](https://github.com/rapidsai/rmm/pull/990)) [@robertmaynard](https://github.com/robertmaynard)
- Skip async mr tests when cuda runtime/driver &lt; 11.2 ([#986](https://github.com/rapidsai/rmm/pull/986)) [@rongou](https://github.com/rongou)
- Fix warning/error in debug assertion in device_uvector.hpp ([#979](https://github.com/rapidsai/rmm/pull/979)) [@harrism](https://github.com/harrism)
- Fix signed/unsigned comparison warning ([#970](https://github.com/rapidsai/rmm/pull/970)) [@jlowe](https://github.com/jlowe)
- Fix comparison of async MRs with different underlying pools. ([#965](https://github.com/rapidsai/rmm/pull/965)) [@harrism](https://github.com/harrism)

## 🚀 New Features

- Use scikit-build for the build process ([#976](https://github.com/rapidsai/rmm/pull/976)) [@vyasr](https://github.com/vyasr)

## 🛠️ Improvements

- Temporarily disable new `ops-bot` functionality ([#1005](https://github.com/rapidsai/rmm/pull/1005)) [@ajschmidt8](https://github.com/ajschmidt8)
- Rename `librmm_tests` to `librmm-tests` ([#1000](https://github.com/rapidsai/rmm/pull/1000)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update `librmm` `conda` recipe ([#997](https://github.com/rapidsai/rmm/pull/997)) [@ajschmidt8](https://github.com/ajschmidt8)
- Remove `no_cma`/`has_cma` variants ([#996](https://github.com/rapidsai/rmm/pull/996)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fix free-before-alloc in multithreaded test ([#992](https://github.com/rapidsai/rmm/pull/992)) [@aladram](https://github.com/aladram)
- Add `.github/ops-bot.yaml` config file ([#991](https://github.com/rapidsai/rmm/pull/991)) [@ajschmidt8](https://github.com/ajschmidt8)
- Log allocation failures ([#988](https://github.com/rapidsai/rmm/pull/988)) [@rongou](https://github.com/rongou)
- Update `librmm` `conda` outputs ([#983](https://github.com/rapidsai/rmm/pull/983)) [@ajschmidt8](https://github.com/ajschmidt8)
- Bump Python requirements in `setup.cfg` and `rmm_dev.yml` ([#982](https://github.com/rapidsai/rmm/pull/982)) [@shwina](https://github.com/shwina)
- New benchmark compares concurrent throughput of device_vector and device_uvector ([#981](https://github.com/rapidsai/rmm/pull/981)) [@harrism](https://github.com/harrism)
- Update `librmm` recipe to output `librmm_tests` package ([#978](https://github.com/rapidsai/rmm/pull/978)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update upload.sh to use `--croot` ([#975](https://github.com/rapidsai/rmm/pull/975)) [@AyodeAwe](https://github.com/AyodeAwe)
- Fix `conda` uploads ([#974](https://github.com/rapidsai/rmm/pull/974)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add CMake `install` rules for tests ([#969](https://github.com/rapidsai/rmm/pull/969)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add device_buffer::ssize() and device_uvector::ssize() ([#966](https://github.com/rapidsai/rmm/pull/966)) [@harrism](https://github.com/harrism)
- Added yml file for cudatoolkit version 11.6 ([#964](https://github.com/rapidsai/rmm/pull/964)) [@alhad-deshpande](https://github.com/alhad-deshpande)
- Replace `ccache` with `sccache` ([#963](https://github.com/rapidsai/rmm/pull/963)) [@ajschmidt8](https://github.com/ajschmidt8)
- Make `pool_memory_resource::pool_size()` public ([#962](https://github.com/rapidsai/rmm/pull/962)) [@shwina](https://github.com/shwina)
- Allow construction of cuda_async_memory_resource from existing pool ([#889](https://github.com/rapidsai/rmm/pull/889)) [@fkallen](https://github.com/fkallen)

# RMM 22.02.00 (2 Feb 2022)

## 🐛 Bug Fixes

- Use numba to get CUDA runtime version. ([#946](https://github.com/rapidsai/rmm/pull/946)) [@bdice](https://github.com/bdice)
- Temporarily disable warnings for unknown pragmas ([#942](https://github.com/rapidsai/rmm/pull/942)) [@harrism](https://github.com/harrism)
- Build benchmarks in RMM CI ([#941](https://github.com/rapidsai/rmm/pull/941)) [@harrism](https://github.com/harrism)
- Headers that use `std::thread` now include &lt;thread&gt; ([#938](https://github.com/rapidsai/rmm/pull/938)) [@robertmaynard](https://github.com/robertmaynard)
- Fix failing stream test with a debug-only death test ([#934](https://github.com/rapidsai/rmm/pull/934)) [@harrism](https://github.com/harrism)
- Prevent `DeviceBuffer` DeviceMemoryResource premature release ([#931](https://github.com/rapidsai/rmm/pull/931)) [@viclafargue](https://github.com/viclafargue)
- Fix failing tracking test ([#929](https://github.com/rapidsai/rmm/pull/929)) [@harrism](https://github.com/harrism)

## 🛠️ Improvements

- Prepare upload scripts for Python 3.7 removal ([#952](https://github.com/rapidsai/rmm/pull/952)) [@Ethyling](https://github.com/Ethyling)
- Fix imports tests syntax ([#935](https://github.com/rapidsai/rmm/pull/935)) [@Ethyling](https://github.com/Ethyling)
- Remove `IncludeCategories` from `.clang-format` ([#933](https://github.com/rapidsai/rmm/pull/933)) [@codereport](https://github.com/codereport)
- Replace use of custom CUDA bindings with CUDA-Python ([#930](https://github.com/rapidsai/rmm/pull/930)) [@shwina](https://github.com/shwina)
- Remove `setup.py` from `update-release.sh` script ([#926](https://github.com/rapidsai/rmm/pull/926)) [@ajschmidt8](https://github.com/ajschmidt8)
- Improve C++ Test Coverage ([#920](https://github.com/rapidsai/rmm/pull/920)) [@harrism](https://github.com/harrism)
- Improve the Arena allocator to reduce memory fragmentation ([#916](https://github.com/rapidsai/rmm/pull/916)) [@rongou](https://github.com/rongou)
- Simplify CMake linting with cmake-format ([#913](https://github.com/rapidsai/rmm/pull/913)) [@vyasr](https://github.com/vyasr)

# RMM 21.12.00 (9 Dec 2021)

## 🚨 Breaking Changes

- Parameterize exception type caught by failure_callback_resource_adaptor ([#898](https://github.com/rapidsai/rmm/pull/898)) [@harrism](https://github.com/harrism)

## 🐛 Bug Fixes

- Update recipes for Enhanced Compatibility ([#910](https://github.com/rapidsai/rmm/pull/910)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fix `librmm` uploads ([#909](https://github.com/rapidsai/rmm/pull/909)) [@ajschmidt8](https://github.com/ajschmidt8)
- Use spdlog/fmt/ostr.h as it supports external fmt library ([#907](https://github.com/rapidsai/rmm/pull/907)) [@robertmaynard](https://github.com/robertmaynard)
- Fix variable names in logging macro calls ([#897](https://github.com/rapidsai/rmm/pull/897)) [@harrism](https://github.com/harrism)
- Keep rapids cmake version in sync ([#876](https://github.com/rapidsai/rmm/pull/876)) [@robertmaynard](https://github.com/robertmaynard)

## 📖 Documentation

- Replace `to_device()` in docs  with `DeviceBuffer.to_device()` ([#902](https://github.com/rapidsai/rmm/pull/902)) [@shwina](https://github.com/shwina)
- Fix return value docs for supports_get_mem_info ([#884](https://github.com/rapidsai/rmm/pull/884)) [@harrism](https://github.com/harrism)

## 🚀 New Features

- Out-of-memory callback resource adaptor ([#892](https://github.com/rapidsai/rmm/pull/892)) [@madsbk](https://github.com/madsbk)

## 🛠️ Improvements

- suppress spurious clang-tidy warnings in debug macros ([#914](https://github.com/rapidsai/rmm/pull/914)) [@rongou](https://github.com/rongou)
- C++ code coverage support ([#905](https://github.com/rapidsai/rmm/pull/905)) [@harrism](https://github.com/harrism)
- Provide ./build.sh flag to control CUDA async malloc support ([#901](https://github.com/rapidsai/rmm/pull/901)) [@robertmaynard](https://github.com/robertmaynard)
- Parameterize exception type caught by failure_callback_resource_adaptor ([#898](https://github.com/rapidsai/rmm/pull/898)) [@harrism](https://github.com/harrism)
- Throw `rmm::out_of_memory` when we know for sure ([#894](https://github.com/rapidsai/rmm/pull/894)) [@rongou](https://github.com/rongou)
- Update `conda` recipes for Enhanced Compatibility effort ([#893](https://github.com/rapidsai/rmm/pull/893)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add functions to query the stream of device_uvector and device_scalar ([#887](https://github.com/rapidsai/rmm/pull/887)) [@fkallen](https://github.com/fkallen)
- Add spdlog to install export set ([#886](https://github.com/rapidsai/rmm/pull/886)) [@trxcllnt](https://github.com/trxcllnt)

# RMM 21.10.00 (7 Oct 2021)

## 🚨 Breaking Changes

- Delete cuda_async_memory_resource copy/move ctors/operators ([#860](https://github.com/rapidsai/rmm/pull/860)) [@jrhemstad](https://github.com/jrhemstad)

## 🐛 Bug Fixes

- Fix parameter name in asserts ([#875](https://github.com/rapidsai/rmm/pull/875)) [@vyasr](https://github.com/vyasr)
- Disallow zero-size stream pools ([#873](https://github.com/rapidsai/rmm/pull/873)) [@harrism](https://github.com/harrism)
- Correct namespace usage in host memory resources ([#872](https://github.com/rapidsai/rmm/pull/872)) [@divyegala](https://github.com/divyegala)
- fix race condition in limiting resource adapter ([#869](https://github.com/rapidsai/rmm/pull/869)) [@rongou](https://github.com/rongou)
- Install the right cudatoolkit in the conda env in gpu/build.sh ([#864](https://github.com/rapidsai/rmm/pull/864)) [@shwina](https://github.com/shwina)
- Disable copy/move ctors and operator= from free_list classes ([#862](https://github.com/rapidsai/rmm/pull/862)) [@harrism](https://github.com/harrism)
- Delete cuda_async_memory_resource copy/move ctors/operators ([#860](https://github.com/rapidsai/rmm/pull/860)) [@jrhemstad](https://github.com/jrhemstad)
- Improve concurrency of stream_ordered_memory_resource by stealing less ([#851](https://github.com/rapidsai/rmm/pull/851)) [@harrism](https://github.com/harrism)
- Use the new RAPIDS.cmake to fetch rapids-cmake ([#838](https://github.com/rapidsai/rmm/pull/838)) [@robertmaynard](https://github.com/robertmaynard)

## 📖 Documentation

- Forward-merge branch-21.08 to branch-21.10 ([#846](https://github.com/rapidsai/rmm/pull/846)) [@jakirkham](https://github.com/jakirkham)

## 🛠️ Improvements

- Forward-merge `branch-21.08` into `branch-21.10` ([#877](https://github.com/rapidsai/rmm/pull/877)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add .clang-tidy and fix clang-tidy warnings ([#857](https://github.com/rapidsai/rmm/pull/857)) [@harrism](https://github.com/harrism)
- Update to use rapids-cmake 21.10 pre-configured packages ([#854](https://github.com/rapidsai/rmm/pull/854)) [@robertmaynard](https://github.com/robertmaynard)
- Clean up: use std::size_t, include cstddef and aligned.hpp where missing ([#852](https://github.com/rapidsai/rmm/pull/852)) [@harrism](https://github.com/harrism)
- tweak the arena mr to reduce fragmentation ([#845](https://github.com/rapidsai/rmm/pull/845)) [@rongou](https://github.com/rongou)
- Fix transitive include in cuda_device header ([#843](https://github.com/rapidsai/rmm/pull/843)) [@wphicks](https://github.com/wphicks)
- Refactor cmake style ([#842](https://github.com/rapidsai/rmm/pull/842)) [@robertmaynard](https://github.com/robertmaynard)
- add multi stream allocations benchmark. ([#841](https://github.com/rapidsai/rmm/pull/841)) [@cwharris](https://github.com/cwharris)
- Enforce default visibility for `get_map`. ([#833](https://github.com/rapidsai/rmm/pull/833)) [@trivialfis](https://github.com/trivialfis)
- ENH Replace gpuci_conda_retry with gpuci_mamba_retry ([#823](https://github.com/rapidsai/rmm/pull/823)) [@dillon-cullinan](https://github.com/dillon-cullinan)
- Execution policy class ([#816](https://github.com/rapidsai/rmm/pull/816)) [@viclafargue](https://github.com/viclafargue)

# RMM 21.08.00 (4 Aug 2021)

## 🚨 Breaking Changes

- Refactor `rmm::device_scalar` in terms of `rmm::device_uvector` ([#789](https://github.com/rapidsai/rmm/pull/789)) [@harrism](https://github.com/harrism)
- Explicit streams in device_buffer ([#775](https://github.com/rapidsai/rmm/pull/775)) [@harrism](https://github.com/harrism)

## 🐛 Bug Fixes

- Pin spdlog in dev conda envs ([#835](https://github.com/rapidsai/rmm/pull/835)) [@trxcllnt](https://github.com/trxcllnt)
- Pinning spdlog because recent updates are causing compile issues. ([#831](https://github.com/rapidsai/rmm/pull/831)) [@cjnolet](https://github.com/cjnolet)
- update isort to 5.6.4 ([#822](https://github.com/rapidsai/rmm/pull/822)) [@cwharris](https://github.com/cwharris)
- fix align_up namespace in aligned_resource_adaptor.hpp ([#820](https://github.com/rapidsai/rmm/pull/820)) [@rongou](https://github.com/rongou)
- Run updated isort hook on pxd files ([#812](https://github.com/rapidsai/rmm/pull/812)) [@charlesbluca](https://github.com/charlesbluca)
- find_package(RMM) can now be called multiple times safely ([#811](https://github.com/rapidsai/rmm/pull/811)) [@robertmaynard](https://github.com/robertmaynard)
- Fix building on CUDA 11.3 ([#809](https://github.com/rapidsai/rmm/pull/809)) [@benfred](https://github.com/benfred)
- Remove leading zeros in version_config.hpp ([#793](https://github.com/rapidsai/rmm/pull/793)) [@hcho3](https://github.com/hcho3)

## 📖 Documentation

- Fix PoolMemoryResource Python doc examples ([#807](https://github.com/rapidsai/rmm/pull/807)) [@harrism](https://github.com/harrism)
- Fix incorrect href in README.md ([#804](https://github.com/rapidsai/rmm/pull/804)) [@benchislett](https://github.com/benchislett)
- Update build instruction in README ([#797](https://github.com/rapidsai/rmm/pull/797)) [@hcho3](https://github.com/hcho3)
- Document compute sanitizer memcheck support ([#790](https://github.com/rapidsai/rmm/pull/790)) [@harrism](https://github.com/harrism)

## 🚀 New Features

- Bump isort, enable Cython package resorting ([#806](https://github.com/rapidsai/rmm/pull/806)) [@charlesbluca](https://github.com/charlesbluca)
- Support multiple output sinks in logging_resource_adaptor ([#791](https://github.com/rapidsai/rmm/pull/791)) [@harrism](https://github.com/harrism)
- Add Statistics Resource Adaptor and cython bindings to `tracking_resource_adaptor` and `statistics_resource_adaptor` ([#626](https://github.com/rapidsai/rmm/pull/626)) [@mdemoret-nv](https://github.com/mdemoret-nv)

## 🛠️ Improvements

- Fix isort in cuda_stream_view.pxd ([#827](https://github.com/rapidsai/rmm/pull/827)) [@harrism](https://github.com/harrism)
- Cython extension for rmm::cuda_stream_pool ([#818](https://github.com/rapidsai/rmm/pull/818)) [@divyegala](https://github.com/divyegala)
- Fix building on cuda 11.4 ([#817](https://github.com/rapidsai/rmm/pull/817)) [@benfred](https://github.com/benfred)
- Updating Clang Version to 11.0.0 ([#814](https://github.com/rapidsai/rmm/pull/814)) [@codereport](https://github.com/codereport)
- Add spdlog to `rmm-exports` if found by CPM ([#810](https://github.com/rapidsai/rmm/pull/810)) [@trxcllnt](https://github.com/trxcllnt)
- Fix `21.08` forward-merge conflicts ([#803](https://github.com/rapidsai/rmm/pull/803)) [@ajschmidt8](https://github.com/ajschmidt8)
- RMM now leverages rapids-cmake to reduce CMake boilerplate ([#800](https://github.com/rapidsai/rmm/pull/800)) [@robertmaynard](https://github.com/robertmaynard)
- Refactor `rmm::device_scalar` in terms of `rmm::device_uvector` ([#789](https://github.com/rapidsai/rmm/pull/789)) [@harrism](https://github.com/harrism)
- make it easier to include rmm in other projects ([#788](https://github.com/rapidsai/rmm/pull/788)) [@rongou](https://github.com/rongou)
- Compile Cython with C++17. ([#787](https://github.com/rapidsai/rmm/pull/787)) [@vyasr](https://github.com/vyasr)
- Fix Merge Conflicts ([#786](https://github.com/rapidsai/rmm/pull/786)) [@ajschmidt8](https://github.com/ajschmidt8)
- Explicit streams in device_buffer ([#775](https://github.com/rapidsai/rmm/pull/775)) [@harrism](https://github.com/harrism)

# RMM 21.06.00 (9 Jun 2021)

## 🐛 Bug Fixes

- FindThrust now guards against multiple inclusion by different consumers ([#784](https://github.com/rapidsai/rmm/pull/784)) [@robertmaynard](https://github.com/robertmaynard)

## 📖 Documentation

- Document synchronization requirements on device_buffer copy ctors ([#772](https://github.com/rapidsai/rmm/pull/772)) [@harrism](https://github.com/harrism)

## 🚀 New Features

- add a resource adapter to align on a specified size ([#768](https://github.com/rapidsai/rmm/pull/768)) [@rongou](https://github.com/rongou)

## 🛠️ Improvements

- Update environment variable used to determine `cuda_version` ([#785](https://github.com/rapidsai/rmm/pull/785)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update `CHANGELOG.md` links for calver ([#781](https://github.com/rapidsai/rmm/pull/781)) [@ajschmidt8](https://github.com/ajschmidt8)
- Merge `branch-0.19` into `branch-21.06` ([#779](https://github.com/rapidsai/rmm/pull/779)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update docs build script ([#776](https://github.com/rapidsai/rmm/pull/776)) [@ajschmidt8](https://github.com/ajschmidt8)
- upgrade spdlog to 1.8.5 ([#658](https://github.com/rapidsai/rmm/pull/658)) [@rongou](https://github.com/rongou)

# RMM 0.19.0 (21 Apr 2021)

## 🚨 Breaking Changes

- Avoid potential race conditions in device_scalar/device_uvector setters ([#725](https://github.com/rapidsai/rmm/pull/725)) [@harrism](https://github.com/harrism)

## 🐛 Bug Fixes

- Fix typo in setup.py ([#746](https://github.com/rapidsai/rmm/pull/746)) [@galipremsagar](https://github.com/galipremsagar)
- Revert &quot;Update `rmm` conda recipe pinning of `librmm`&quot; ([#743](https://github.com/rapidsai/rmm/pull/743)) [@raydouglass](https://github.com/raydouglass)
- Update `rmm` conda recipe pinning of `librmm` ([#738](https://github.com/rapidsai/rmm/pull/738)) [@mike-wendt](https://github.com/mike-wendt)
- RMM doesn&#39;t require the CUDA language to be enabled by consumers ([#737](https://github.com/rapidsai/rmm/pull/737)) [@robertmaynard](https://github.com/robertmaynard)
- Fix setup.py to work in a non-conda environment setup ([#733](https://github.com/rapidsai/rmm/pull/733)) [@galipremsagar](https://github.com/galipremsagar)
- Fix auto-detecting GPU architectures ([#727](https://github.com/rapidsai/rmm/pull/727)) [@trxcllnt](https://github.com/trxcllnt)
- CMAKE_CUDA_ARCHITECTURES doesn&#39;t change when build-system invokes cmake ([#726](https://github.com/rapidsai/rmm/pull/726)) [@robertmaynard](https://github.com/robertmaynard)
- Ship memory_resource_wrappers.hpp as package_data ([#715](https://github.com/rapidsai/rmm/pull/715)) [@shwina](https://github.com/shwina)
- Only include SetGPUArchs in the top-level CMakeLists.txt ([#713](https://github.com/rapidsai/rmm/pull/713)) [@trxcllnt](https://github.com/trxcllnt)
- Fix unknown CMake command &quot;CPMFindPackage&quot; ([#699](https://github.com/rapidsai/rmm/pull/699)) [@standbyme](https://github.com/standbyme)

## 📖 Documentation

- Fix host_memory_resource signature typo ([#728](https://github.com/rapidsai/rmm/pull/728)) [@miguelusque](https://github.com/miguelusque)

## 🚀 New Features

- Clarify log file name behaviour in docs ([#722](https://github.com/rapidsai/rmm/pull/722)) [@shwina](https://github.com/shwina)
- Add Cython definitions for device_uvector ([#720](https://github.com/rapidsai/rmm/pull/720)) [@shwina](https://github.com/shwina)
- Python bindings for `cuda_async_memory_resource` ([#718](https://github.com/rapidsai/rmm/pull/718)) [@shwina](https://github.com/shwina)

## 🛠️ Improvements

- Fix cython tests ([#749](https://github.com/rapidsai/rmm/pull/749)) [@galipremsagar](https://github.com/galipremsagar)
- Add requirements for rmm ([#739](https://github.com/rapidsai/rmm/pull/739)) [@galipremsagar](https://github.com/galipremsagar)
- device_uvector can be used within thrust::optional ([#734](https://github.com/rapidsai/rmm/pull/734)) [@robertmaynard](https://github.com/robertmaynard)
- arena_memory_resource optimization: disable tracking allocated blocks by default ([#732](https://github.com/rapidsai/rmm/pull/732)) [@rongou](https://github.com/rongou)
- Remove CMAKE_CURRENT_BINARY_DIR path in rmm&#39;s target_include_directories ([#731](https://github.com/rapidsai/rmm/pull/731)) [@trxcllnt](https://github.com/trxcllnt)
- set CMAKE_CUDA_ARCHITECTURES to OFF instead of undefined ([#729](https://github.com/rapidsai/rmm/pull/729)) [@trxcllnt](https://github.com/trxcllnt)
- Avoid potential race conditions in device_scalar/device_uvector setters ([#725](https://github.com/rapidsai/rmm/pull/725)) [@harrism](https://github.com/harrism)
- Update Changelog Link ([#723](https://github.com/rapidsai/rmm/pull/723)) [@ajschmidt8](https://github.com/ajschmidt8)
- Prepare Changelog for Automation ([#717](https://github.com/rapidsai/rmm/pull/717)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update 0.18 changelog entry ([#716](https://github.com/rapidsai/rmm/pull/716)) [@ajschmidt8](https://github.com/ajschmidt8)
- Simplify cmake cuda architectures handling ([#709](https://github.com/rapidsai/rmm/pull/709)) [@robertmaynard](https://github.com/robertmaynard)
- Build only `compute` for the newest arch in CMAKE_CUDA_ARCHITECTURES ([#706](https://github.com/rapidsai/rmm/pull/706)) [@robertmaynard](https://github.com/robertmaynard)
- ENH Build with Ninja &amp; Pass ccache variables to conda recipe ([#705](https://github.com/rapidsai/rmm/pull/705)) [@dillon-cullinan](https://github.com/dillon-cullinan)
- pool_memory_resource optimization: disable tracking allocated blocks by default ([#702](https://github.com/rapidsai/rmm/pull/702)) [@harrism](https://github.com/harrism)
- Allow the build directory of rmm to be used for `find_package(rmm)` ([#698](https://github.com/rapidsai/rmm/pull/698)) [@robertmaynard](https://github.com/robertmaynard)
- Adds a linear accessor to RMM cuda stream pool ([#696](https://github.com/rapidsai/rmm/pull/696)) [@afender](https://github.com/afender)
- Fix merge conflicts for #692 ([#694](https://github.com/rapidsai/rmm/pull/694)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fix merge conflicts for #692 ([#693](https://github.com/rapidsai/rmm/pull/693)) [@ajschmidt8](https://github.com/ajschmidt8)
- Remove C++ Wrappers in `memory_resource_adaptors.hpp` Needed by Cython ([#662](https://github.com/rapidsai/rmm/pull/662)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Improve Cython Lifetime Management by Adding References in `DeviceBuffer` ([#661](https://github.com/rapidsai/rmm/pull/661)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Add support for streams in CuPy allocator ([#654](https://github.com/rapidsai/rmm/pull/654)) [@pentschev](https://github.com/pentschev)

# RMM 0.18.0 (24 Feb 2021)

## Breaking Changes 🚨

- Remove DeviceBuffer synchronization on default stream (#650) @pentschev
- Add a Stream class that wraps CuPy/Numba/CudaStream (#636) @shwina

## Bug Fixes 🐛

- SetGPUArchs updated to work around a CMake FindCUDAToolkit issue (#695) @robertmaynard
- Remove duplicate conda build command (#670) @raydouglass
- Update CMakeLists.txt VERSION to 0.18.0 (#665) @trxcllnt
- Fix wrong attribute names leading to DEBUG log build issues (#653) @pentschev

## Documentation 📖

- Correct inconsistencies in README and CONTRIBUTING docs (#682) @robertmaynard
- Enable tag generation for doxygen (#672) @ajschmidt8
- Document that `managed_memory_resource` does not work with NVIDIA vGPU (#656) @harrism

## New Features 🚀

- Enabling/disabling logging after initialization (#678) @shwina
- `cuda_async_memory_resource` built on `cudaMallocAsync` (#676) @harrism
- Create labeler.yml (#669) @jolorunyomi
- Expose the version string in C++ and Python (#666) @hcho3
- Add a CUDA stream pool (#659) @harrism
- Add a Stream class that wraps CuPy/Numba/CudaStream (#636) @shwina

## Improvements 🛠️

- Update stale GHA with exemptions &amp; new labels (#707) @mike-wendt
- Add GHA to mark issues/prs as stale/rotten (#700) @Ethyling
- Auto-label PRs based on their content (#691) @ajschmidt8
- Prepare Changelog for Automation (#688) @ajschmidt8
- Build.sh use cmake --build to drive build system invocation (#686) @robertmaynard
- Fix failed automerge (#683) @harrism
- Auto-label PRs based on their content (#681) @jolorunyomi
- Build RMM tests/benchmarks with -Wall flag (#674) @trxcllnt
- Remove DeviceBuffer synchronization on default stream (#650) @pentschev
- Simplify `rmm::exec_policy` and refactor Thrust support (#647) @harrism

# RMM 0.17.0 (10 Dec 2020)

## New Features

- PR #609 Adds `polymorphic_allocator` and `stream_allocator_adaptor`
- PR #596 Add `tracking_memory_resource_adaptor` to help catch memory leaks
- PR #608 Add stream wrapper type
- PR #632 Add RMM Python docs

## Improvements

- PR #604 CMake target cleanup, formatting, linting
- PR #599 Make the arena memory resource work better with the producer/consumer mode
- PR #612 Drop old Python `device_array*` API
- PR #603 Always test both legacy and per-thread default stream
- PR #611 Add a note to the contribution guide about requiring 2 C++ reviewers
- PR #615 Improve gpuCI Scripts
- PR #627 Cleanup gpuCI Scripts
- PR #635 Add Python docs build to gpuCI

## Bug Fixes

- PR #592 Add `auto_flush` to `make_logging_adaptor`
- PR #602 Fix `device_scalar` and its tests so that they use the correct CUDA stream
- PR #621 Make `rmm::cuda_stream_default` a `constexpr`
- PR #625 Use `librmm` conda artifact when building `rmm` conda package
- PR #631 Force local conda artifact install
- PR #634 Fix conda uploads
- PR #639 Fix release script version updater based on CMake reformatting
- PR #641 Fix adding "LANGUAGES" after version number in CMake in release script


# RMM 0.16.0 (21 Oct 2020)

## New Features

- PR #529 Add debug logging and fix multithreaded replay benchmark
- PR #560 Remove deprecated `get/set_default_resource` APIs
- PR #543 Add an arena-based memory resource
- PR #580 Install CMake config with RMM
- PR #591 Allow the replay bench to simulate different GPU memory sizes
- PR #594 Adding limiting memory resource adaptor

## Improvements

- PR #474 Use CMake find_package(CUDAToolkit)
- PR #477 Just use `None` for `strides` in `DeviceBuffer`
- PR #528 Add maximum_pool_size parameter to reinitialize API
- PR #532 Merge free lists in pool_memory_resource to defragment before growing from upstream
- PR #537 Add CMake option to disable deprecation warnings
- PR #541 Refine CMakeLists.txt to make it easy to import by external projects
- PR #538 Upgrade CUB and Thrust to the latest commits
- PR #542 Pin conda spdlog versions to 1.7.0
- PR #550 Remove CXX11 ABI handling from CMake
- PR #578 Switch thrust to use the NVIDIA/thrust repo
- PR #553 CMake cleanup
- PR #556 By default, don't create a debug log file unless there are warnings/errors
- PR #561 Remove CNMeM and make RMM header-only
- PR #565 CMake: Simplify gtest/gbench handling
- PR #566 CMake: use CPM for thirdparty dependencies
- PR #568 Upgrade googletest to v1.10.0
- PR #572 CMake: prefer locally installed thirdparty packages
- PR #579 CMake: handle thrust via target
- PR #581 Improve logging documentation
- PR #585 Update ci/local/README.md
- PR #587 Replaced `move` with `std::move`
- PR #588 Use installed C++ RMM in python build
- PR #601 Make maximum pool size truly optional (grow until failure)

## Bug Fixes

- PR #545 Fix build to support using `clang` as the host compiler
- PR #534 Fix `pool_memory_resource` failure when init and max pool sizes are equal
- PR #546 Remove CUDA driver linking and correct NVTX macro.
- PR #569 Correct `device_scalar::set_value` to pass host value by reference to avoid copying from invalid value
- PR #559 Fix `align_down` to only change unaligned values.
- PR #577 Fix CMake `LOGGING_LEVEL` issue which caused verbose logging / performance regression.
- PR #582 Fix handling of per-thread default stream when not compiled for PTDS
- PR #590 Add missing `CODE_OF_CONDUCT.md`
- PR #595 Fix pool_mr example in README.md


# RMM 0.15.0 (26 Aug 2020)

## New Features

- PR #375 Support out-of-band buffers in Python pickling
- PR #391 Add `get_default_resource_type`
- PR #396 Remove deprecated RMM APIs
- PR #425 Add CUDA per-thread default stream support and thread safety to `pool_memory_resource`
- PR #436 Always build and test with per-thread default stream enabled in the GPU CI build
- PR #444 Add `owning_wrapper` to simplify lifetime management of resources and their upstreams
- PR #449 Stream-ordered suballocator base class and per-thread default stream support
          and thread safety for `fixed_size_memory_resource`
- PR #450 Add support for new build process (Project Flash)
- PR #457 New `binning_memory_resource` (replaces `hybrid_memory_resource` and
          `fixed_multisize_memory_resource`).
- PR #458 Add `get/set_per_device_resource` to better support multi-GPU per process applications
- PR #466 Deprecate CNMeM.
- PR #489 Move `cudf._cuda` into `rmm._cuda`
- PR #504 Generate `gpu.pxd` based on cuda version as a preprocessor step
- PR #506 Upload rmm package per version python-cuda combo

## Improvements

- PR #428 Add the option to automatically flush memory allocate/free logs
- PR #378 Use CMake `FetchContent` to obtain latest release of `cub` and `thrust`
- PR #377 A better way to fetch `spdlog`
- PR #372 Use CMake `FetchContent` to obtain `cnmem` instead of git submodule
- PR #382 Rely on NumPy arrays for out-of-band pickling
- PR #386 Add short commit to conda package name
- PR #401 Update `get_ipc_handle()` to use cuda driver API
- PR #404 Make all memory resources thread safe in Python
- PR #402 Install dependencies via rapids-build-env
- PR #405 Move doc customization scripts to Jenkins
- PR #427 Add DeviceBuffer.release() cdef method
- PR #414 Add element-wise access for device_uvector
- PR #421 Capture thread id in logging and improve logger testing
- PR #426 Added multi-threaded support to replay benchmark
- PR #429 Fix debug build and add new CUDA assert utility
- PR #435 Update conda upload versions for new supported CUDA/Python
- PR #437 Test with `pickle5` (for older Python versions)
- PR #443 Remove thread safe adaptor from PoolMemoryResource
- PR #445 Make all resource operators/ctors explicit
- PR #447 Update Python README with info about DeviceBuffer/MemoryResource and external libraries
- PR #456 Minor cleanup: always use rmm/-prefixed includes
- PR #461 cmake improvements to be more target-based
- PR #468 update past release dates in changelog
- PR #486 Document relationship between active CUDA devices and resources
- PR #493 Rely on C++ lazy Memory Resource initialization behavior instead of initializing in Python

## Bug Fixes

- PR #433 Fix python imports
- PR #400 Fix segfault in RANDOM_ALLOCATIONS_BENCH
- PR #383 Explicitly require NumPy
- PR #398 Fix missing head flag in merge_blocks (pool_memory_resource) and improve block class
- PR #403 Mark Cython `memory_resource_wrappers` `extern` as `nogil`
- PR #406 Sets Google Benchmark to a fixed version, v1.5.1.
- PR #434 Fix issue with incorrect docker image being used in local build script
- PR #463 Revert cmake change for cnmem header not being added to source directory
- PR #464 More completely revert cnmem.h cmake changes
- PR #473 Fix initialization logic in pool_memory_resource
- PR #479 Fix usage of block printing in pool_memory_resource
- PR #490 Allow importing RMM without initializing CUDA driver
- PR #484 Fix device_uvector copy constructor compilation error and add test
- PR #498 Max pool growth less greedy
- PR #500 Use tempfile rather than hardcoded path in `test_rmm_csv_log`
- PR #511 Specify `--basetemp` for `py.test` run
- PR #509 Fix missing : before __LINE__ in throw string of RMM_CUDA_TRY
- PR #510 Fix segfault in pool_memory_resource when a CUDA stream is destroyed
- PR #525 Patch Thrust to workaround `CUDA_CUB_RET_IF_FAIL` macro clearing CUDA errors


# RMM 0.14.0 (03 Jun 2020)

## New Features

- PR #317 Provide External Memory Management Plugin for Numba
- PR #362 Add spdlog as a dependency in the conda package
- PR #360 Support logging to stdout/stderr
- PR #341 Enable logging
- PR #343 Add in option to statically link against cudart
- PR #364 Added new uninitialized device vector type, `device_uvector`

## Improvements

- PR #369 Use CMake `FetchContent` to obtain `spdlog` instead of vendoring
- PR #366 Remove installation of extra test dependencies
- PR #354 Add CMake option for per-thread default stream
- PR #350 Add .clang-format file & format all files
- PR #358 Fix typo in `rmm_cupy_allocator` docstring
- PR #357 Add Docker 19 support to local gpuci build
- PR #365 Make .clang-format consistent with cuGRAPH and cuDF
- PR #371 Add docs build script to repository
- PR #363 Expose `memory_resources` in Python

## Bug Fixes

- PR #373 Fix build.sh
- PR #346 Add clearer exception message when RMM_LOG_FILE is unset
- PR #347 Mark rmmFinalizeWrapper nogil
- PR #348 Fix unintentional use of pool-managed resource.
- PR #367 Fix flake8 issues
- PR #368 Fix `clang-format` missing comma bug
- PR #370 Fix stream and mr use in `device_buffer` methods
- PR #379 Remove deprecated calls from synchronization.cpp
- PR #381 Remove test_benchmark.cpp from cmakelists
- PR #392 SPDLOG matches other header-only acquisition patterns


# RMM 0.13.0 (31 Mar 2020)

## New Features

- PR #253 Add `frombytes` to convert `bytes`-like to `DeviceBuffer`
- PR #252 Add `__sizeof__` method to `DeviceBuffer`
- PR #258 Define pickling behavior for `DeviceBuffer`
- PR #261 Add `__bytes__` method to `DeviceBuffer`
- PR #262 Moved device memory resource files to `mr/device` directory
- PR #266 Drop `rmm.auto_device`
- PR #268 Add Cython/Python `copy_to_host` and `to_device`
- PR #272 Add `host_memory_resource`.
- PR #273 Moved device memory resource tests to `device/` directory.
- PR #274 Add `copy_from_host` method to `DeviceBuffer`
- PR #275 Add `copy_from_device` method to `DeviceBuffer`
- PR #283 Add random allocation benchmark.
- PR #287 Enabled CUDA CXX11 for unit tests.
- PR #292 Revamped RMM exceptions.
- PR #297 Use spdlog to implement `logging_resource_adaptor`.
- PR #303 Added replay benchmark.
- PR #319 Add `thread_safe_resource_adaptor` class.
- PR #314 New suballocator memory_resources.
- PR #330 Fixed incorrect name of `stream_free_blocks_` debug symbol.
- PR #331 Move to C++14 and deprecate legacy APIs.

## Improvements

- PR #246 Type `DeviceBuffer` arguments to `__cinit__`
- PR #249 Use `DeviceBuffer` in `device_array`
- PR #255 Add standard header to all Cython files
- PR #256 Cast through `uintptr_t` to `cudaStream_t`
- PR #254 Use `const void*` in `DeviceBuffer.__cinit__`
- PR #257 Mark Cython-exposed C++ functions that raise
- PR #269 Doc sync behavior in `copy_ptr_to_host`
- PR #278 Allocate a `bytes` object to fill up with RMM log data
- PR #280 Drop allocation/deallocation of `offset`
- PR #282 `DeviceBuffer` use default constructor for size=0
- PR #296 Use CuPy's `UnownedMemory` for RMM-backed allocations
- PR #310 Improve `device_buffer` allocation logic.
- PR #309 Sync default stream in `DeviceBuffer` constructor
- PR #326 Sync only on copy construction
- PR #308 Fix typo in README
- PR #334 Replace `rmm_allocator` for Thrust allocations
- PR #345 Remove stream synchronization from `device_scalar` constructor and `set_value`

## Bug Fixes

- PR #298 Remove RMM_CUDA_TRY from cuda_event_timer destructor
- PR #299 Fix assert condition blocking debug builds
- PR #300 Fix host mr_tests compile error
- PR #312 Fix libcudf compilation errors due to explicit defaulted device_buffer constructor


# RMM 0.12.0 (04 Feb 2020)

## New Features

- PR #218 Add `_DevicePointer`
- PR #219 Add method to copy `device_buffer` back to host memory
- PR #222 Expose free and total memory in Python interface
- PR #235 Allow construction of `DeviceBuffer` with a `stream`

## Improvements

- PR #214 Add codeowners
- PR #226 Add some tests of the Python `DeviceBuffer`
- PR #233 Reuse the same `CUDA_HOME` logic from cuDF
- PR #234 Add missing `size_t` in `DeviceBuffer`
- PR #239 Cleanup `DeviceBuffer`'s `__cinit__`
- PR #242 Special case 0-size `DeviceBuffer` in `tobytes`
- PR #244 Explicitly force `DeviceBuffer.size` to an `int`
- PR #247 Simplify casting in `tobytes` and other cleanup

## Bug Fixes

- PR #215 Catch polymorphic exceptions by reference instead of by value
- PR #221 Fix segfault calling rmmGetInfo when uninitialized
- PR #225 Avoid invoking Python operations in c_free
- PR #230 Fix duplicate symbol issues with `copy_to_host`
- PR #232 Move `copy_to_host` doc back to header file


# RMM 0.11.0 (11 Dec 2019)

## New Features

- PR #106 Added multi-GPU initialization
- PR #167 Added value setter to `device_scalar`
- PR #163 Add Cython bindings to `device_buffer`
- PR #177 Add `__cuda_array_interface__` to `DeviceBuffer`
- PR #198 Add `rmm.rmm_cupy_allocator()`

## Improvements

- PR #161 Use `std::atexit` to finalize RMM after Python interpreter shutdown
- PR #165 Align memory resource allocation sizes to 8-byte
- PR #171 Change public API of RMM to only expose `reinitialize(...)`
- PR #175 Drop `cython` from run requirements
- PR #169 Explicit stream argument for device_buffer methods
- PR #186 Add nbytes and len to DeviceBuffer
- PR #188 Require kwargs in `DeviceBuffer`'s constructor
- PR #194 Drop unused imports from `device_buffer.pyx`
- PR #196 Remove unused CUDA conda labels
- PR #200 Simplify DeviceBuffer methods

## Bug Fixes

- PR #174 Make `device_buffer` default ctor explicit to work around type_dispatcher issue in libcudf.
- PR #170 Always build librmm and rmm, but conditionally upload based on CUDA / Python version
- PR #182 Prefix `DeviceBuffer`'s C functions
- PR #189 Drop `__reduce__` from `DeviceBuffer`
- PR #193 Remove thrown exception from `rmm_allocator::deallocate`
- PR #224 Slice the CSV log before converting to bytes


# RMM 0.10.0 (16 Oct 2019)

## New Features

- PR #99 Added `device_buffer` class
- PR #133 Added `device_scalar` class

## Improvements

- PR #123 Remove driver install from ci scripts
- PR #131 Use YYMMDD tag in nightly build
- PR #137 Replace CFFI python bindings with Cython
- PR #127 Use Memory Resource classes for allocations

## Bug Fixes

- PR #107 Fix local build generated file ownerships
- PR #110 Fix Skip Test Functionality
- PR #125 Fixed order of private variables in LogIt
- PR #139 Expose `_make_finalizer` python API needed by cuDF
- PR #142 Fix ignored exceptions in Cython
- PR #146 Fix rmmFinalize() not freeing memory pools
- PR #149 Force finalization of RMM objects before RMM is finalized (Python)
- PR #154 Set ptr to 0 on rmm::alloc error
- PR #157 Check if initialized before freeing for Numba finalizer and use `weakref` instead of `atexit`


# RMM 0.9.0 (21 Aug 2019)

## New Features

- PR #96 Added `device_memory_resource` for beginning of overhaul of RMM design
- PR #103 Add and use unified build script

## Improvements

- PR #111 Streamline CUDA_REL environment variable
- PR #113 Handle ucp.BufferRegion objects in auto_device

## Bug Fixes

   ...


# RMM 0.8.0 (27 June 2019)

## New Features

- PR #95 Add skip test functionality to build.sh

## Improvements

   ...

## Bug Fixes

- PR #92 Update docs version


# RMM 0.7.0 (10 May 2019)

## New Features

- PR #67 Add random_allocate microbenchmark in tests/performance
- PR #70 Create conda environments and conda recipes
- PR #77 Add local build script to mimic gpuCI
- PR #80 Add build script for docs

## Improvements

- PR #76 Add cudatoolkit conda dependency
- PR #84 Use latest release version in update-version CI script
- PR #90 Avoid using c++14 auto return type for thrust_rmm_allocator.h

## Bug Fixes

- PR #68 Fix signed/unsigned mismatch in random_allocate benchmark
- PR #74 Fix rmm conda recipe librmm version pinning
- PR #72 Remove unnecessary _BSD_SOURCE define in random_allocate.cpp


# RMM 0.6.0 (18 Mar 2019)

## New Features

- PR #43 Add gpuCI build & test scripts
- PR #44 Added API to query whether RMM is initialized and with what options.
- PR #60 Default to CXX11_ABI=ON

## Improvements

## Bug Fixes

- PR #58 Eliminate unreliable check for change in available memory in test
- PR #49 Fix pep8 style errors detected by flake8


# RMM 0.5.0 (28 Jan 2019)

## New Features

- PR #2 Added CUDA Managed Memory allocation mode

## Improvements

- PR #12 Enable building RMM as a submodule
- PR #13 CMake: Added CXX11ABI option and removed Travis references
- PR #16 CMake: Added PARALLEL_LEVEL environment variable handling for GTest build parallelism (matches cuDF)
- PR #17 Update README with v0.5 changes including Managed Memory support

## Bug Fixes

- PR #10 Change cnmem submodule URL to use https
- PR #15 Temporarily disable hanging AllocateTB test for managed memory
- PR #28 Fix invalid reference to local stack variable in `rmm::exec_policy`


# RMM 0.4.0 (20 Dec 2018)

## New Features

- PR #1 Spun off RMM from cuDF into its own repository.

## Improvements

- CUDF PR #472 RMM: Created centralized rmm::device_vector alias and rmm::exec_policy
- CUDF PR #465 Added templated C++ API for RMM to avoid explicit cast to `void**`

## Bug Fixes


RMM was initially implemented as part of cuDF, so we include the relevant changelog history below.

# cuDF 0.3.0 (23 Nov 2018)

## New Features

- PR #336 CSV Reader string support

## Improvements

- CUDF PR #333 Add Rapids Memory Manager documentation
- CUDF PR #321 Rapids Memory Manager adds file/line location logging and convenience macros

## Bug Fixes


# cuDF 0.2.0 and cuDF 0.1.0

These were initial releases of cuDF based on previously separate pyGDF and libGDF libraries. RMM was initially implemented as part of libGDF.
