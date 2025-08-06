# rmm 25.08.00 (6 Aug 2025)

## 🚨 Breaking Changes

- Update requirements to CUDA 12.0+ ([#1984](https://github.com/rapidsai/rmm/pull/1984)) [@bdice](https://github.com/bdice)
- Remove CUDA 11 from dependencies.yaml ([#1934](https://github.com/rapidsai/rmm/pull/1934)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- stop uploading packages to downloads.rapids.ai ([#1929](https://github.com/rapidsai/rmm/pull/1929)) [@jameslamb](https://github.com/jameslamb)

## 🐛 Bug Fixes

- Temporarily disable failing test on HMM systems. ([#1950](https://github.com/rapidsai/rmm/pull/1950)) [@bdice](https://github.com/bdice)
- Fix race conditions and deadlocks in REPLAY_BENCH ([#1940](https://github.com/rapidsai/rmm/pull/1940)) [@wence-](https://github.com/wence-)

## 📖 Documentation

- Update Python build instructions to include librmm wheel ([#1978](https://github.com/rapidsai/rmm/pull/1978)) [@gmarkall](https://github.com/gmarkall)
- Fix Python path in CONTRIBUTING.md ([#1936](https://github.com/rapidsai/rmm/pull/1936)) [@bdice](https://github.com/bdice)

## 🚀 New Features

- Update requirements to CUDA 12.0+ ([#1984](https://github.com/rapidsai/rmm/pull/1984)) [@bdice](https://github.com/bdice)

## 🛠️ Improvements

- Use size_type in device_uvector ([#1992](https://github.com/rapidsai/rmm/pull/1992)) [@bdice](https://github.com/bdice)
- chore: remove unused line from update-version.sh ([#1989](https://github.com/rapidsai/rmm/pull/1989)) [@gforsyth](https://github.com/gforsyth)
- Revert &quot;Update branches that trigger nightlies ([#1954)&quot; (#1988](https://github.com/rapidsai/rmm/pull/1954)&quot; (#1988)) [@gforsyth](https://github.com/gforsyth)
- fix(docker): use versioned `-latest` tag for all `rapidsai` images ([#1987](https://github.com/rapidsai/rmm/pull/1987)) [@gforsyth](https://github.com/gforsyth)
- Move more implementations to precompiled shared library ([#1980](https://github.com/rapidsai/rmm/pull/1980)) [@bdice](https://github.com/bdice)
- [pre-commit.ci] pre-commit autoupdate ([#1979](https://github.com/rapidsai/rmm/pull/1979)) [@pre-commit-ci[bot]](https://github.com/pre-commit-ci[bot])
- Add managed memory resource to replay benchmark ([#1938](https://github.com/rapidsai/rmm/pull/1938)) [@pentschev](https://github.com/pentschev)
- Remove CUDA 11 from dependencies.yaml ([#1934](https://github.com/rapidsai/rmm/pull/1934)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Remove CUDA 11 devcontainers and update CI scripts ([#1933](https://github.com/rapidsai/rmm/pull/1933)) [@bdice](https://github.com/bdice)
- refactor(rattler): remove cuda11 options and general cleanup ([#1932](https://github.com/rapidsai/rmm/pull/1932)) [@gforsyth](https://github.com/gforsyth)
- stop uploading packages to downloads.rapids.ai ([#1929](https://github.com/rapidsai/rmm/pull/1929)) [@jameslamb](https://github.com/jameslamb)
- Forward-merge branch-25.06 into branch-25.08 ([#1925](https://github.com/rapidsai/rmm/pull/1925)) [@gforsyth](https://github.com/gforsyth)
- Branch 25.08 merge branch 25.06 ([#1914](https://github.com/rapidsai/rmm/pull/1914)) [@vyasr](https://github.com/vyasr)
- Forward-merge branch-25.06 into branch-25.08 ([#1905](https://github.com/rapidsai/rmm/pull/1905)) [@gforsyth](https://github.com/gforsyth)

# rmm 25.06.00 (5 Jun 2025)

## 🚨 Breaking Changes

- Convert part of RMM to a precompiled library ([#1896](https://github.com/rapidsai/rmm/pull/1896)) [@bdice](https://github.com/bdice)
- Move RMM C++ code into cpp directory. ([#1883](https://github.com/rapidsai/rmm/pull/1883)) [@bdice](https://github.com/bdice)

## 🐛 Bug Fixes

- Run system MR tests in isolation. ([#1945](https://github.com/rapidsai/rmm/pull/1945)) [@bdice](https://github.com/bdice)
- Use auditwheel to properly retag the wheel ([#1913](https://github.com/rapidsai/rmm/pull/1913)) [@vyasr](https://github.com/vyasr)
- Fix logger macros ([#1884](https://github.com/rapidsai/rmm/pull/1884)) [@vyasr](https://github.com/vyasr)

## 📖 Documentation

- Move docs to top level. ([#1917](https://github.com/rapidsai/rmm/pull/1917)) [@bdice](https://github.com/bdice)
- Update Readme for the logging `set_level` ([#1911](https://github.com/rapidsai/rmm/pull/1911)) [@JigaoLuo](https://github.com/JigaoLuo)
- Fixed documentation example for `DeviceBuffer.to_device` ([#1881](https://github.com/rapidsai/rmm/pull/1881)) [@TomAugspurger](https://github.com/TomAugspurger)

## 🚀 New Features

- Convert part of RMM to a precompiled library ([#1896](https://github.com/rapidsai/rmm/pull/1896)) [@bdice](https://github.com/bdice)
- Set mempool hw_decompress flag if driver supports it ([#1875](https://github.com/rapidsai/rmm/pull/1875)) [@bdice](https://github.com/bdice)
- Expose option to enable fabric memory handle support to Python ([#1787](https://github.com/rapidsai/rmm/pull/1787)) [@pentschev](https://github.com/pentschev)

## 🛠️ Improvements

- fix(pytest): disable warning that gets raised to INTERNALERROR in pytest8.4.0 ([#1942](https://github.com/rapidsai/rmm/pull/1942)) [@gforsyth](https://github.com/gforsyth)
- use &#39;rapids-init-pip&#39; in wheel CI, other CI changes ([#1926](https://github.com/rapidsai/rmm/pull/1926)) [@jameslamb](https://github.com/jameslamb)
- Finish CUDA 12.9 migration and use branch-25.06 workflows ([#1921](https://github.com/rapidsai/rmm/pull/1921)) [@bdice](https://github.com/bdice)
- Update to clang 20 ([#1918](https://github.com/rapidsai/rmm/pull/1918)) [@bdice](https://github.com/bdice)
- Quote head_rev in conda recipes ([#1915](https://github.com/rapidsai/rmm/pull/1915)) [@bdice](https://github.com/bdice)
- Build and test with CUDA 12.9.0 ([#1907](https://github.com/rapidsai/rmm/pull/1907)) [@bdice](https://github.com/bdice)
- Fix cpp wheel name to librmm. ([#1903](https://github.com/rapidsai/rmm/pull/1903)) [@bdice](https://github.com/bdice)
- Revert &quot;Publish wheels and conda packages from Github Artifacts&quot; ([#1898](https://github.com/rapidsai/rmm/pull/1898)) [@bdice](https://github.com/bdice)
- Publish wheels and conda packages from Github Artifacts ([#1897](https://github.com/rapidsai/rmm/pull/1897)) [@VenkateshJaya](https://github.com/VenkateshJaya)
- Download build artifacts from Github for CI ([#1895](https://github.com/rapidsai/rmm/pull/1895)) [@VenkateshJaya](https://github.com/VenkateshJaya)
- remove mkdir and test corresponding shared workflow ([#1892](https://github.com/rapidsai/rmm/pull/1892)) [@msarahan](https://github.com/msarahan)
- Revert &quot;Auto-sync draft PRs&quot; ([#1891](https://github.com/rapidsai/rmm/pull/1891)) [@bdice](https://github.com/bdice)
- Auto-sync draft PRs ([#1890](https://github.com/rapidsai/rmm/pull/1890)) [@bdice](https://github.com/bdice)
- Add ARM conda environments ([#1889](https://github.com/rapidsai/rmm/pull/1889)) [@bdice](https://github.com/bdice)
- Vendor RAPIDS.cmake to avoid network call. ([#1886](https://github.com/rapidsai/rmm/pull/1886)) [@bdice](https://github.com/bdice)
- [pre-commit.ci] pre-commit autoupdate ([#1885](https://github.com/rapidsai/rmm/pull/1885)) [@pre-commit-ci[bot]](https://github.com/pre-commit-ci[bot])
- Move RMM C++ code into cpp directory. ([#1883](https://github.com/rapidsai/rmm/pull/1883)) [@bdice](https://github.com/bdice)
- refactor(rattler): enable strict channel priority for builds ([#1867](https://github.com/rapidsai/rmm/pull/1867)) [@gforsyth](https://github.com/gforsyth)
- Add support for Python 3.13 ([#1851](https://github.com/rapidsai/rmm/pull/1851)) [@bdice](https://github.com/bdice)
- Streamlining wheel builds to use fixed location and uploading build artifacts to Github ([#1810](https://github.com/rapidsai/rmm/pull/1810)) [@VenkateshJaya](https://github.com/VenkateshJaya)

# rmm 25.04.00 (9 Apr 2025)

## 🚨 Breaking Changes

- Add OOM fail reason, attempted allocation size to exception messages (retry) ([#1844](https://github.com/rapidsai/rmm/pull/1844)) [@pmattione-nvidia](https://github.com/pmattione-nvidia)
- Use new rapids-logger library ([#1808](https://github.com/rapidsai/rmm/pull/1808)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- Revert &quot;Set mempool hw_decompress flag if driver supports it ([#1854)&quot; (#1873](https://github.com/rapidsai/rmm/pull/1854)&quot; (#1873)) [@wence-](https://github.com/wence-)
- Fix run export on cudatoolkit ([#1862](https://github.com/rapidsai/rmm/pull/1862)) [@vyasr](https://github.com/vyasr)
- Fix dependencies.yaml for update-version.sh ([#1859](https://github.com/rapidsai/rmm/pull/1859)) [@raydouglass](https://github.com/raydouglass)
- Embed `__FILE__` as C-string for prefix replacement ([#1858](https://github.com/rapidsai/rmm/pull/1858)) [@jakirkham](https://github.com/jakirkham)
- Add OOM fail reason, attempted allocation size to exception messages (retry) ([#1844](https://github.com/rapidsai/rmm/pull/1844)) [@pmattione-nvidia](https://github.com/pmattione-nvidia)
- Revert &quot;Add OOM fail reason, attempted allocation size to exception messages&quot; ([#1843](https://github.com/rapidsai/rmm/pull/1843)) [@pmattione-nvidia](https://github.com/pmattione-nvidia)
- fix GITHUB_WORKSPACE not being present locally ([#1841](https://github.com/rapidsai/rmm/pull/1841)) [@msarahan](https://github.com/msarahan)
- Add telemetry setup to build workflows ([#1838](https://github.com/rapidsai/rmm/pull/1838)) [@bdice](https://github.com/bdice)
- Use static gbench ([#1837](https://github.com/rapidsai/rmm/pull/1837)) [@bdice](https://github.com/bdice)
- Fixes for rattler recipe ([#1835](https://github.com/rapidsai/rmm/pull/1835)) [@bdice](https://github.com/bdice)
- Depend on rapids-logger in host to prevent redistribution ([#1834](https://github.com/rapidsai/rmm/pull/1834)) [@bdice](https://github.com/bdice)
- Add OOM fail reason, attempted allocation size to exception messages ([#1827](https://github.com/rapidsai/rmm/pull/1827)) [@pmattione-nvidia](https://github.com/pmattione-nvidia)

## 📖 Documentation

- mr/host: fix incorrect docs usage of device_memory_resource ([#1809](https://github.com/rapidsai/rmm/pull/1809)) [@ghost](https://github.com/ghost)

## 🚀 New Features

- Add async view memory resource bindings to Python. ([#1864](https://github.com/rapidsai/rmm/pull/1864)) [@bdice](https://github.com/bdice)
- Run examples in CI ([#1850](https://github.com/rapidsai/rmm/pull/1850)) [@bdice](https://github.com/bdice)
- Add tests for RMM internal macros. ([#1847](https://github.com/rapidsai/rmm/pull/1847)) [@bdice](https://github.com/bdice)
- Add basic example. ([#1800](https://github.com/rapidsai/rmm/pull/1800)) [@bdice](https://github.com/bdice)

## 🛠️ Improvements

- Set mempool hw_decompress flag if driver supports it ([#1854](https://github.com/rapidsai/rmm/pull/1854)) [@wence-](https://github.com/wence-)
- Error if LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE is not defined. ([#1852](https://github.com/rapidsai/rmm/pull/1852)) [@bdice](https://github.com/bdice)
- Fix for -fdebug-prefix-map breaking sccache ([#1846](https://github.com/rapidsai/rmm/pull/1846)) [@bdice](https://github.com/bdice)
- fix(rattler): force `cuda_major` and `date_string` to be strings ([#1842](https://github.com/rapidsai/rmm/pull/1842)) [@gforsyth](https://github.com/gforsyth)
- use gha-tools rapids-telemetry-setup for mkdir -p ([#1839](https://github.com/rapidsai/rmm/pull/1839)) [@msarahan](https://github.com/msarahan)
- fix(rattler): resolve all overlinking errors ([#1836](https://github.com/rapidsai/rmm/pull/1836)) [@gforsyth](https://github.com/gforsyth)
- Update rattler-build recipe with assorted small fixes ([#1832](https://github.com/rapidsai/rmm/pull/1832)) [@gforsyth](https://github.com/gforsyth)
- Sccache stats telemetry ([#1830](https://github.com/rapidsai/rmm/pull/1830)) [@msarahan](https://github.com/msarahan)
- Consolidate more Conda solves in CI ([#1828](https://github.com/rapidsai/rmm/pull/1828)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Require CMake 3.30.4 ([#1826](https://github.com/rapidsai/rmm/pull/1826)) [@robertmaynard](https://github.com/robertmaynard)
- Create Conda CI test env in one step ([#1824](https://github.com/rapidsai/rmm/pull/1824)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Apply IWYU changes and fix deprecated GTest usage ([#1821](https://github.com/rapidsai/rmm/pull/1821)) [@bdice](https://github.com/bdice)
- Remove unnecessary index ([#1820](https://github.com/rapidsai/rmm/pull/1820)) [@vyasr](https://github.com/vyasr)
- Use shared-workflows branch-25.04 ([#1816](https://github.com/rapidsai/rmm/pull/1816)) [@bdice](https://github.com/bdice)
- Use `rapids-pip-retry` in CI jobs that might need retries ([#1814](https://github.com/rapidsai/rmm/pull/1814)) [@gforsyth](https://github.com/gforsyth)
- Use nightly matrix for branch tests. ([#1813](https://github.com/rapidsai/rmm/pull/1813)) [@bdice](https://github.com/bdice)
- Use build_type input ([#1812](https://github.com/rapidsai/rmm/pull/1812)) [@bdice](https://github.com/bdice)
- Add `build_type` to workflow inputs ([#1811](https://github.com/rapidsai/rmm/pull/1811)) [@gforsyth](https://github.com/gforsyth)
- Use new rapids-logger library ([#1808](https://github.com/rapidsai/rmm/pull/1808)) [@vyasr](https://github.com/vyasr)
- Forward-merge branch-25.02 to branch-25.04 ([#1806](https://github.com/rapidsai/rmm/pull/1806)) [@bdice](https://github.com/bdice)
- disallow fallback to Make in wheel builds ([#1804](https://github.com/rapidsai/rmm/pull/1804)) [@jameslamb](https://github.com/jameslamb)
- Migrate to NVKS for amd64 CI runners ([#1803](https://github.com/rapidsai/rmm/pull/1803)) [@bdice](https://github.com/bdice)
- Branch 25.04 merge branch 25.02 ([#1799](https://github.com/rapidsai/rmm/pull/1799)) [@vyasr](https://github.com/vyasr)
- Port to rattler-build ([#1796](https://github.com/rapidsai/rmm/pull/1796)) [@gforsyth](https://github.com/gforsyth)

# rmm 25.02.00 (13 Feb 2025)

## 🚨 Breaking Changes

- Switch to using separate rapids-logger repo ([#1774](https://github.com/rapidsai/rmm/pull/1774)) [@vyasr](https://github.com/vyasr)
- Remove deprecated factory functions from resource adaptors. ([#1767](https://github.com/rapidsai/rmm/pull/1767)) [@bdice](https://github.com/bdice)
- Remove `rmm._lib` ([#1765](https://github.com/rapidsai/rmm/pull/1765)) [@Matt711](https://github.com/Matt711)
- Remove memory access flags from cuda_async_memory_resource ([#1754](https://github.com/rapidsai/rmm/pull/1754)) [@abellina](https://github.com/abellina)
- Create logger wrapper around spdlog that can be easily reused in other libraries ([#1722](https://github.com/rapidsai/rmm/pull/1722)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- Add missing array header include ([#1771](https://github.com/rapidsai/rmm/pull/1771)) [@robertmaynard](https://github.com/robertmaynard)
- Remove memory access flags from cuda_async_memory_resource ([#1754](https://github.com/rapidsai/rmm/pull/1754)) [@abellina](https://github.com/abellina)
- Update build.sh ([#1749](https://github.com/rapidsai/rmm/pull/1749)) [@vyasr](https://github.com/vyasr)
- Fix some logger issues ([#1739](https://github.com/rapidsai/rmm/pull/1739)) [@vyasr](https://github.com/vyasr)
- Use consistent signature for target_link_libraries ([#1738](https://github.com/rapidsai/rmm/pull/1738)) [@vyasr](https://github.com/vyasr)

## 📖 Documentation

- Revise README. ([#1747](https://github.com/rapidsai/rmm/pull/1747)) [@bdice](https://github.com/bdice)

## 🚀 New Features

- Make the stream module a part of the public API ([#1775](https://github.com/rapidsai/rmm/pull/1775)) [@Matt711](https://github.com/Matt711)
- Remove deprecated factory functions from resource adaptors. ([#1767](https://github.com/rapidsai/rmm/pull/1767)) [@bdice](https://github.com/bdice)
- Remove `rmm._lib` ([#1765](https://github.com/rapidsai/rmm/pull/1765)) [@Matt711](https://github.com/Matt711)
- Reduce dependencies on numba. ([#1761](https://github.com/rapidsai/rmm/pull/1761)) [@bdice](https://github.com/bdice)
- Use ruff, remove isort and black. ([#1759](https://github.com/rapidsai/rmm/pull/1759)) [@bdice](https://github.com/bdice)
- Use bindings layout for all cuda-python imports. ([#1756](https://github.com/rapidsai/rmm/pull/1756)) [@bdice](https://github.com/bdice)
- Add configuration for pre-commit.ci, update pre-commit hooks ([#1746](https://github.com/rapidsai/rmm/pull/1746)) [@bdice](https://github.com/bdice)
- Adds fabric handle and memory protection flags to cuda_async_memory_resource ([#1743](https://github.com/rapidsai/rmm/pull/1743)) [@abellina](https://github.com/abellina)
- Remove upper bounds on cuda-python to allow 12.6.2 and 11.8.5 ([#1729](https://github.com/rapidsai/rmm/pull/1729)) [@bdice](https://github.com/bdice)

## 🛠️ Improvements

- Revert CUDA 12.8 shared workflow branch changes ([#1805](https://github.com/rapidsai/rmm/pull/1805)) [@vyasr](https://github.com/vyasr)
- Build and test with CUDA 12.8.0 ([#1797](https://github.com/rapidsai/rmm/pull/1797)) [@bdice](https://github.com/bdice)
- Disable exec checks for `device_uvector::operator=` ([#1790](https://github.com/rapidsai/rmm/pull/1790)) [@miscco](https://github.com/miscco)
- Add upper bound to prevent usage of numba 0.61.0 ([#1789](https://github.com/rapidsai/rmm/pull/1789)) [@galipremsagar](https://github.com/galipremsagar)
- Add shellcheck to pre-commit and fix warnings ([#1788](https://github.com/rapidsai/rmm/pull/1788)) [@gforsyth](https://github.com/gforsyth)
- Add spdlog back as a requirement for now ([#1780](https://github.com/rapidsai/rmm/pull/1780)) [@vyasr](https://github.com/vyasr)
- [pre-commit.ci] pre-commit autoupdate ([#1778](https://github.com/rapidsai/rmm/pull/1778)) [@pre-commit-ci[bot]](https://github.com/pre-commit-ci[bot])
- Use rapids-cmake for the logger ([#1776](https://github.com/rapidsai/rmm/pull/1776)) [@vyasr](https://github.com/vyasr)
- Switch to using separate rapids-logger repo ([#1774](https://github.com/rapidsai/rmm/pull/1774)) [@vyasr](https://github.com/vyasr)
- Use GCC 13 in CUDA 12 conda builds. ([#1773](https://github.com/rapidsai/rmm/pull/1773)) [@bdice](https://github.com/bdice)
- Check if nightlies have succeeded recently enough ([#1772](https://github.com/rapidsai/rmm/pull/1772)) [@vyasr](https://github.com/vyasr)
- Fix codespell behavior. ([#1769](https://github.com/rapidsai/rmm/pull/1769)) [@bdice](https://github.com/bdice)
- Remove ignored cuda-python deprecation warning. ([#1768](https://github.com/rapidsai/rmm/pull/1768)) [@bdice](https://github.com/bdice)
- Forward-merge branch-24.12 to branch-25.02 ([#1766](https://github.com/rapidsai/rmm/pull/1766)) [@bdice](https://github.com/bdice)
- Update version references in workflow ([#1757](https://github.com/rapidsai/rmm/pull/1757)) [@AyodeAwe](https://github.com/AyodeAwe)
- gate telemetry dispatch calls on TELEMETRY_ENABLED env var ([#1752](https://github.com/rapidsai/rmm/pull/1752)) [@msarahan](https://github.com/msarahan)
- Update cuda-python lower bounds to 12.6.2 / 11.8.5 ([#1751](https://github.com/rapidsai/rmm/pull/1751)) [@bdice](https://github.com/bdice)
- remove certs and simplify telemetry summarize ([#1750](https://github.com/rapidsai/rmm/pull/1750)) [@msarahan](https://github.com/msarahan)
- stop installing &#39;wheel&#39; in wheel-building script ([#1748](https://github.com/rapidsai/rmm/pull/1748)) [@jameslamb](https://github.com/jameslamb)
- Require approval to run CI on draft PRs ([#1737](https://github.com/rapidsai/rmm/pull/1737)) [@bdice](https://github.com/bdice)
- Create logger wrapper around spdlog that can be easily reused in other libraries ([#1722](https://github.com/rapidsai/rmm/pull/1722)) [@vyasr](https://github.com/vyasr)
- Add breaking change workflow trigger ([#1719](https://github.com/rapidsai/rmm/pull/1719)) [@AyodeAwe](https://github.com/AyodeAwe)

# rmm 24.12.00 (11 Dec 2024)

## 🚨 Breaking Changes

- Deprecate support for directly accessing logger ([#1690](https://github.com/rapidsai/rmm/pull/1690)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- Query total memory in failure_callback_resource_adaptor tests ([#1734](https://github.com/rapidsai/rmm/pull/1734)) [@harrism](https://github.com/harrism)
- Treat deprecation warnings as errors and fix deprecation warnings in replay benchmark ([#1728](https://github.com/rapidsai/rmm/pull/1728)) [@harrism](https://github.com/harrism)
- Disallow cuda-python 12.6.1 and 11.8.4 ([#1720](https://github.com/rapidsai/rmm/pull/1720)) [@bdice](https://github.com/bdice)
- Fix typos in .gitignore ([#1697](https://github.com/rapidsai/rmm/pull/1697)) [@charlesbluca](https://github.com/charlesbluca)
- Fix `rmm ._lib` imports ([#1693](https://github.com/rapidsai/rmm/pull/1693)) [@Matt711](https://github.com/Matt711)

## 📖 Documentation

- Fix docs warning ([#1706](https://github.com/rapidsai/rmm/pull/1706)) [@bdice](https://github.com/bdice)
- Update cross-link to cuda-python object ([#1699](https://github.com/rapidsai/rmm/pull/1699)) [@wence-](https://github.com/wence-)

## 🚀 New Features

- Correct rmm tests for validity of device pointers ([#1714](https://github.com/rapidsai/rmm/pull/1714)) [@robertmaynard](https://github.com/robertmaynard)
- Update rmm tests to use rapids_cmake_support_conda_env ([#1707](https://github.com/rapidsai/rmm/pull/1707)) [@robertmaynard](https://github.com/robertmaynard)
- adding telemetry ([#1692](https://github.com/rapidsai/rmm/pull/1692)) [@msarahan](https://github.com/msarahan)
- Make `cudaMallocAsync` logic non-optional as we require CUDA 11.2+ ([#1667](https://github.com/rapidsai/rmm/pull/1667)) [@robertmaynard](https://github.com/robertmaynard)

## 🛠️ Improvements

- enforce wheel size limits, README formatting in CI ([#1726](https://github.com/rapidsai/rmm/pull/1726)) [@jameslamb](https://github.com/jameslamb)
- Remove all explicit usage of fmtlib ([#1724](https://github.com/rapidsai/rmm/pull/1724)) [@harrism](https://github.com/harrism)
- WIP: put a ceiling on cuda-python ([#1723](https://github.com/rapidsai/rmm/pull/1723)) [@jameslamb](https://github.com/jameslamb)
- use rapids-generate-pip-constraints to pin to oldest dependencies in CI ([#1716](https://github.com/rapidsai/rmm/pull/1716)) [@jameslamb](https://github.com/jameslamb)
- Deprecate `rmm._lib` ([#1713](https://github.com/rapidsai/rmm/pull/1713)) [@Matt711](https://github.com/Matt711)
- print sccache stats in builds ([#1712](https://github.com/rapidsai/rmm/pull/1712)) [@jameslamb](https://github.com/jameslamb)
- [fea] Expose the arena mr to the Python interface. ([#1711](https://github.com/rapidsai/rmm/pull/1711)) [@trivialfis](https://github.com/trivialfis)
- devcontainer: replace `VAULT_HOST` with `AWS_ROLE_ARN` ([#1708](https://github.com/rapidsai/rmm/pull/1708)) [@jjacobelli](https://github.com/jjacobelli)
- make conda installs in CI stricter (part 2) ([#1703](https://github.com/rapidsai/rmm/pull/1703)) [@jameslamb](https://github.com/jameslamb)
- Add BUILD_SHARED_LIBS option defaulting to ON ([#1702](https://github.com/rapidsai/rmm/pull/1702)) [@wence-](https://github.com/wence-)
- make conda installs in CI stricter ([#1696](https://github.com/rapidsai/rmm/pull/1696)) [@jameslamb](https://github.com/jameslamb)
- Prune workflows based on changed files ([#1695](https://github.com/rapidsai/rmm/pull/1695)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Deprecate support for directly accessing logger ([#1690](https://github.com/rapidsai/rmm/pull/1690)) [@vyasr](https://github.com/vyasr)
- Use `rmm::percent_of_free_device_memory` in arena test ([#1689](https://github.com/rapidsai/rmm/pull/1689)) [@wence-](https://github.com/wence-)
- exclude &#39;gcovr&#39; from list of development pip packages ([#1688](https://github.com/rapidsai/rmm/pull/1688)) [@jameslamb](https://github.com/jameslamb)
- [Improvement] Reorganize Cython to separate C++ bindings and make Cython classes public ([#1676](https://github.com/rapidsai/rmm/pull/1676)) [@Matt711](https://github.com/Matt711)

# rmm 24.10.00 (9 Oct 2024)

## 🚨 Breaking Changes

- Inline functions that return static references must have default visibility ([#1653](https://github.com/rapidsai/rmm/pull/1653)) [@wence-](https://github.com/wence-)
- Hide visibility of non-public symbols ([#1644](https://github.com/rapidsai/rmm/pull/1644)) [@jameslamb](https://github.com/jameslamb)
- Deprecate adaptor factories. ([#1626](https://github.com/rapidsai/rmm/pull/1626)) [@bdice](https://github.com/bdice)

## 🐛 Bug Fixes

- Add missing include to `resource_ref.hpp` ([#1677](https://github.com/rapidsai/rmm/pull/1677)) [@miscco](https://github.com/miscco)
- Remove the friend declaration with an attribute ([#1669](https://github.com/rapidsai/rmm/pull/1669)) [@kingcrimsontianyu](https://github.com/kingcrimsontianyu)
- Fix `build.sh clean` to delete python build directory ([#1658](https://github.com/rapidsai/rmm/pull/1658)) [@rongou](https://github.com/rongou)
- Stream synchronize before deallocating SAM ([#1655](https://github.com/rapidsai/rmm/pull/1655)) [@rongou](https://github.com/rongou)
- Explicitly mark RMM headers with `RMM_EXPORT` ([#1654](https://github.com/rapidsai/rmm/pull/1654)) [@robertmaynard](https://github.com/robertmaynard)
- Inline functions that return static references must have default visibility ([#1653](https://github.com/rapidsai/rmm/pull/1653)) [@wence-](https://github.com/wence-)
- Use `tool.scikit-build.cmake.version` ([#1637](https://github.com/rapidsai/rmm/pull/1637)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)

## 📖 Documentation

- Recommend `miniforge` for conda install. ([#1681](https://github.com/rapidsai/rmm/pull/1681)) [@bdice](https://github.com/bdice)
- Fix docs cross reference in DeviceBuffer.prefetch ([#1636](https://github.com/rapidsai/rmm/pull/1636)) [@bdice](https://github.com/bdice)

## 🚀 New Features

- [FEA] Allow setting `*_pool_size` with human-readable string ([#1670](https://github.com/rapidsai/rmm/pull/1670)) [@Matt711](https://github.com/Matt711)
- Update RMM adaptors, containers and tests to use get/set_current_device_resource_ref() ([#1661](https://github.com/rapidsai/rmm/pull/1661)) [@harrism](https://github.com/harrism)
- Deprecate adaptor factories. ([#1626](https://github.com/rapidsai/rmm/pull/1626)) [@bdice](https://github.com/bdice)
- Allow testing of earliest/latest dependencies ([#1613](https://github.com/rapidsai/rmm/pull/1613)) [@seberg](https://github.com/seberg)
- Add resource_ref versions of get/set_current_device_resource ([#1598](https://github.com/rapidsai/rmm/pull/1598)) [@harrism](https://github.com/harrism)

## 🛠️ Improvements

- Update update-version.sh to use packaging lib ([#1685](https://github.com/rapidsai/rmm/pull/1685)) [@AyodeAwe](https://github.com/AyodeAwe)
- Use CI workflow branch &#39;branch-24.10&#39; again ([#1683](https://github.com/rapidsai/rmm/pull/1683)) [@jameslamb](https://github.com/jameslamb)
- Update fmt (to 11.0.2) and spdlog (to 1.14.1). ([#1678](https://github.com/rapidsai/rmm/pull/1678)) [@jameslamb](https://github.com/jameslamb)
- Attempt to address oom failures in test suite ([#1672](https://github.com/rapidsai/rmm/pull/1672)) [@wence-](https://github.com/wence-)
- Add support for Python 3.12 ([#1666](https://github.com/rapidsai/rmm/pull/1666)) [@jameslamb](https://github.com/jameslamb)
- Update rapidsai/pre-commit-hooks ([#1663](https://github.com/rapidsai/rmm/pull/1663)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Drop Python 3.9 support ([#1659](https://github.com/rapidsai/rmm/pull/1659)) [@jameslamb](https://github.com/jameslamb)
- Remove NumPy &lt;2 pin ([#1650](https://github.com/rapidsai/rmm/pull/1650)) [@seberg](https://github.com/seberg)
- Hide visibility of non-public symbols ([#1644](https://github.com/rapidsai/rmm/pull/1644)) [@jameslamb](https://github.com/jameslamb)
- Update pre-commit hooks ([#1643](https://github.com/rapidsai/rmm/pull/1643)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Improve update-version.sh ([#1640](https://github.com/rapidsai/rmm/pull/1640)) [@bdice](https://github.com/bdice)
- Install headers into `${CMAKE_INSTALL_INCLUDEDIR}` ([#1633](https://github.com/rapidsai/rmm/pull/1633)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Merge branch-24.08 into branch-24.10 ([#1631](https://github.com/rapidsai/rmm/pull/1631)) [@jameslamb](https://github.com/jameslamb)

# rmm 24.08.00 (7 Aug 2024)

## 🚨 Breaking Changes

- Add a stack to the statistics resource ([#1563](https://github.com/rapidsai/rmm/pull/1563)) [@madsbk](https://github.com/madsbk)

## 🐛 Bug Fixes

- Rename `.devcontainer`s for CUDA 12.5 ([#1615](https://github.com/rapidsai/rmm/pull/1615)) [@jakirkham](https://github.com/jakirkham)
- Avoid accessing statistics_resource_adaptor stack top if it is empty ([#1588](https://github.com/rapidsai/rmm/pull/1588)) [@harrism](https://github.com/harrism)
- Avoid `--find-links`. ([#1583](https://github.com/rapidsai/rmm/pull/1583)) [@bdice](https://github.com/bdice)
- Fix test_python matrix ([#1579](https://github.com/rapidsai/rmm/pull/1579)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Allow anonymous user in devcontainer name ([#1576](https://github.com/rapidsai/rmm/pull/1576)) [@bdice](https://github.com/bdice)

## 📖 Documentation

- Instruct to create associated issue in PR template. ([#1624](https://github.com/rapidsai/rmm/pull/1624)) [@harrism](https://github.com/harrism)
- add rapids-build-backend to docs ([#1614](https://github.com/rapidsai/rmm/pull/1614)) [@jameslamb](https://github.com/jameslamb)
- Revert &quot;Remove HTML builds of librmm ([#1415)&quot; (#1604](https://github.com/rapidsai/rmm/pull/1415)&quot; (#1604)) [@bdice](https://github.com/bdice)
- Add documentation for CPM usage ([#1600](https://github.com/rapidsai/rmm/pull/1600)) [@pauleonix](https://github.com/pauleonix)
- Update Thrust CMake Guide link in README.md ([#1593](https://github.com/rapidsai/rmm/pull/1593)) [@pauleonix](https://github.com/pauleonix)

## 🚀 New Features

- Prefetch resource adaptor ([#1608](https://github.com/rapidsai/rmm/pull/1608)) [@bdice](https://github.com/bdice)
- Add python wrapper for system memory resource ([#1605](https://github.com/rapidsai/rmm/pull/1605)) [@rongou](https://github.com/rongou)
- Refactor mr_ref_tests to not depend on MR base classes ([#1589](https://github.com/rapidsai/rmm/pull/1589)) [@harrism](https://github.com/harrism)
- Add system memory resource ([#1581](https://github.com/rapidsai/rmm/pull/1581)) [@rongou](https://github.com/rongou)
- Add rmm::prefetch() and  DeviceBuffer.prefetch() ([#1573](https://github.com/rapidsai/rmm/pull/1573)) [@harrism](https://github.com/harrism)

## 🛠️ Improvements

- split up CUDA-suffixed dependencies in dependencies.yaml ([#1627](https://github.com/rapidsai/rmm/pull/1627)) [@jameslamb](https://github.com/jameslamb)
- Remove prefetch factory. ([#1625](https://github.com/rapidsai/rmm/pull/1625)) [@bdice](https://github.com/bdice)
- Use workflow branch 24.08 again ([#1617](https://github.com/rapidsai/rmm/pull/1617)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Build and test with CUDA 12.5.1 ([#1607](https://github.com/rapidsai/rmm/pull/1607)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- skip CMake 3.30.0 ([#1603](https://github.com/rapidsai/rmm/pull/1603)) [@jameslamb](https://github.com/jameslamb)
- Add RMM_USE_NVTX cmake option to provide localized control of NVTX for RMM ([#1602](https://github.com/rapidsai/rmm/pull/1602)) [@jlowe](https://github.com/jlowe)
- Use verify-alpha-spec hook ([#1601](https://github.com/rapidsai/rmm/pull/1601)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Avoid --find-links in wheel jobs ([#1586](https://github.com/rapidsai/rmm/pull/1586)) [@jameslamb](https://github.com/jameslamb)
- resolve dependency-file-generator warning, remove unnecessary rapids-build-backend configuration ([#1582](https://github.com/rapidsai/rmm/pull/1582)) [@jameslamb](https://github.com/jameslamb)
- Remove THRUST_WRAPPED_NAMESPACE and tests ([#1578](https://github.com/rapidsai/rmm/pull/1578)) [@harrism](https://github.com/harrism)
- Remove text builds of documentation ([#1575](https://github.com/rapidsai/rmm/pull/1575)) [@vyasr](https://github.com/vyasr)
- ensure update-version.sh preserves alpha specs ([#1572](https://github.com/rapidsai/rmm/pull/1572)) [@jameslamb](https://github.com/jameslamb)
- Add `available_device_memory` to fetch free amount of memory on a GPU ([#1567](https://github.com/rapidsai/rmm/pull/1567)) [@galipremsagar](https://github.com/galipremsagar)
- Add a stack to the statistics resource ([#1563](https://github.com/rapidsai/rmm/pull/1563)) [@madsbk](https://github.com/madsbk)
- Use rapids-build-backend. ([#1502](https://github.com/rapidsai/rmm/pull/1502)) [@bdice](https://github.com/bdice)

# rmm 24.06.00 (5 Jun 2024)

## 🚨 Breaking Changes

- Refactor polymorphic allocator to use device_async_resource_ref ([#1555](https://github.com/rapidsai/rmm/pull/1555)) [@harrism](https://github.com/harrism)
- Remove deprecated functionality ([#1537](https://github.com/rapidsai/rmm/pull/1537)) [@harrism](https://github.com/harrism)
- Remove deprecated cuda_async_memory_resource constructor that takes thrust::optional parameters ([#1535](https://github.com/rapidsai/rmm/pull/1535)) [@harrism](https://github.com/harrism)
- Remove deprecated supports_streams and get_mem_info methods. ([#1519](https://github.com/rapidsai/rmm/pull/1519)) [@harrism](https://github.com/harrism)

## 🐛 Bug Fixes

- rmm needs to link to nvtx3::nvtx3-cpp to support installed nvtx3 ([#1569](https://github.com/rapidsai/rmm/pull/1569)) [@robertmaynard](https://github.com/robertmaynard)
- Make sure rmm wheel dependency on librmm is updated [skip ci] ([#1565](https://github.com/rapidsai/rmm/pull/1565)) [@raydouglass](https://github.com/raydouglass)
- Don&#39;t ignore GCC-specific warning under Clang ([#1557](https://github.com/rapidsai/rmm/pull/1557)) [@aaronmondal](https://github.com/aaronmondal)
- Add publish jobs for C++ wheels ([#1554](https://github.com/rapidsai/rmm/pull/1554)) [@vyasr](https://github.com/vyasr)
- Explicitly use the current device resource in DeviceBuffer ([#1514](https://github.com/rapidsai/rmm/pull/1514)) [@wence-](https://github.com/wence-)

## 📖 Documentation

- Allow specifying mr in DeviceBuffer construction, and document ownership requirements in Python/C++ interfacing ([#1552](https://github.com/rapidsai/rmm/pull/1552)) [@wence-](https://github.com/wence-)
- Fix Python install instruction ([#1547](https://github.com/rapidsai/rmm/pull/1547)) [@wence-](https://github.com/wence-)
- Update multi-gpu discussion for device_buffer and device_vector dtors ([#1524](https://github.com/rapidsai/rmm/pull/1524)) [@wence-](https://github.com/wence-)
- Fix ordering / heading levels in README.md and python example in guide.md ([#1513](https://github.com/rapidsai/rmm/pull/1513)) [@harrism](https://github.com/harrism)

## 🚀 New Features

- Add NVTX support and RMM_FUNC_RANGE() macro ([#1558](https://github.com/rapidsai/rmm/pull/1558)) [@harrism](https://github.com/harrism)
- Always use a static gtest ([#1532](https://github.com/rapidsai/rmm/pull/1532)) [@robertmaynard](https://github.com/robertmaynard)
- Build C++ wheel ([#1529](https://github.com/rapidsai/rmm/pull/1529)) [@vyasr](https://github.com/vyasr)
- Remove deprecated supports_streams and get_mem_info methods. ([#1519](https://github.com/rapidsai/rmm/pull/1519)) [@harrism](https://github.com/harrism)

## 🛠️ Improvements

- update copyright dates ([#1564](https://github.com/rapidsai/rmm/pull/1564)) [@jameslamb](https://github.com/jameslamb)
- Overhaul ops-codeowners ([#1561](https://github.com/rapidsai/rmm/pull/1561)) [@raydouglass](https://github.com/raydouglass)
- Adding support for cupy.cuda.stream.ExternalStream ([#1559](https://github.com/rapidsai/rmm/pull/1559)) [@lilohuang](https://github.com/lilohuang)
- Refactor polymorphic allocator to use device_async_resource_ref ([#1555](https://github.com/rapidsai/rmm/pull/1555)) [@harrism](https://github.com/harrism)
- add RAPIDS copyright pre-commit hook ([#1553](https://github.com/rapidsai/rmm/pull/1553)) [@jameslamb](https://github.com/jameslamb)
- Enable warnings as errors for Python tests ([#1551](https://github.com/rapidsai/rmm/pull/1551)) [@mroeschke](https://github.com/mroeschke)
- Remove header existence tests. ([#1550](https://github.com/rapidsai/rmm/pull/1550)) [@bdice](https://github.com/bdice)
- Only use functions in the limited API ([#1545](https://github.com/rapidsai/rmm/pull/1545)) [@vyasr](https://github.com/vyasr)
- Migrate to `{{ stdlib(&quot;c&quot;) }}` ([#1543](https://github.com/rapidsai/rmm/pull/1543)) [@hcho3](https://github.com/hcho3)
- Fix `cuda11.8` nvcc dependency ([#1542](https://github.com/rapidsai/rmm/pull/1542)) [@trxcllnt](https://github.com/trxcllnt)
- add --rm and --name to devcontainer run args ([#1539](https://github.com/rapidsai/rmm/pull/1539)) [@trxcllnt](https://github.com/trxcllnt)
- Remove deprecated functionality ([#1537](https://github.com/rapidsai/rmm/pull/1537)) [@harrism](https://github.com/harrism)
- Remove deprecated cuda_async_memory_resource constructor that takes thrust::optional parameters ([#1535](https://github.com/rapidsai/rmm/pull/1535)) [@harrism](https://github.com/harrism)
- Make thrust_allocator deallocate safe in multi-device setting ([#1533](https://github.com/rapidsai/rmm/pull/1533)) [@wence-](https://github.com/wence-)
- Move rmm Python package to subdirectory ([#1526](https://github.com/rapidsai/rmm/pull/1526)) [@vyasr](https://github.com/vyasr)
- Remove a file not being used ([#1521](https://github.com/rapidsai/rmm/pull/1521)) [@galipremsagar](https://github.com/galipremsagar)
- Remove unneeded `update-version.sh` update ([#1520](https://github.com/rapidsai/rmm/pull/1520)) [@AyodeAwe](https://github.com/AyodeAwe)
- Enable all tests for `arm` arch ([#1510](https://github.com/rapidsai/rmm/pull/1510)) [@galipremsagar](https://github.com/galipremsagar)

# RMM 24.04.00 (10 Apr 2024)

## 🚨 Breaking Changes

- Accept stream argument in DeviceMemoryResource allocate/deallocate ([#1494](https://github.com/rapidsai/rmm/pull/1494)) [@wence-](https://github.com/wence-)
- Replace all internal usage of `get_upstream` with `get_upstream_resource` ([#1491](https://github.com/rapidsai/rmm/pull/1491)) [@miscco](https://github.com/miscco)
- Deprecate rmm::mr::device_memory_resource::supports_streams() ([#1452](https://github.com/rapidsai/rmm/pull/1452)) [@harrism](https://github.com/harrism)
- Remove deprecated rmm::detail::available_device_memory ([#1438](https://github.com/rapidsai/rmm/pull/1438)) [@harrism](https://github.com/harrism)
- Make device_memory_resource::supports_streams() not pure virtual. Remove derived implementations and calls in RMM ([#1437](https://github.com/rapidsai/rmm/pull/1437)) [@harrism](https://github.com/harrism)
- Deprecate rmm::mr::device_memory_resource::get_mem_info() and supports_get_mem_info(). ([#1436](https://github.com/rapidsai/rmm/pull/1436)) [@harrism](https://github.com/harrism)

## 🐛 Bug Fixes

- Fix search path for torch allocator in editable installs and ensure CUDA support is available ([#1498](https://github.com/rapidsai/rmm/pull/1498)) [@vyasr](https://github.com/vyasr)
- Accept stream argument in DeviceMemoryResource allocate/deallocate ([#1494](https://github.com/rapidsai/rmm/pull/1494)) [@wence-](https://github.com/wence-)
- Run STATISTICS_TEST and TRACKING_TEST in serial to avoid OOM errors. ([#1487](https://github.com/rapidsai/rmm/pull/1487)) [@bdice](https://github.com/bdice)

## 📖 Documentation

- Pin to recent breathe, to prevent getting an unsupported sphinx version. ([#1495](https://github.com/rapidsai/rmm/pull/1495)) [@bdice](https://github.com/bdice)

## 🚀 New Features

- Replace all internal usage of `get_upstream` with `get_upstream_resource` ([#1491](https://github.com/rapidsai/rmm/pull/1491)) [@miscco](https://github.com/miscco)
- Add complete set of resource ref aliases ([#1479](https://github.com/rapidsai/rmm/pull/1479)) [@nvdbaranec](https://github.com/nvdbaranec)
- Automate include grouping using clang-format ([#1463](https://github.com/rapidsai/rmm/pull/1463)) [@harrism](https://github.com/harrism)
- Add `get_upstream_resource` to resource adaptors ([#1456](https://github.com/rapidsai/rmm/pull/1456)) [@miscco](https://github.com/miscco)
- Deprecate rmm::mr::device_memory_resource::supports_streams() ([#1452](https://github.com/rapidsai/rmm/pull/1452)) [@harrism](https://github.com/harrism)
- Remove duplicated memory_resource_tests ([#1451](https://github.com/rapidsai/rmm/pull/1451)) [@miscco](https://github.com/miscco)
- Change `rmm::exec_policy` to take `async_resource_ref` ([#1449](https://github.com/rapidsai/rmm/pull/1449)) [@miscco](https://github.com/miscco)
- Change `device_scalar` to take `async_resource_ref` ([#1447](https://github.com/rapidsai/rmm/pull/1447)) [@miscco](https://github.com/miscco)
- Add device_async_resource_ref convenience alias ([#1441](https://github.com/rapidsai/rmm/pull/1441)) [@harrism](https://github.com/harrism)
- Remove deprecated rmm::detail::available_device_memory ([#1438](https://github.com/rapidsai/rmm/pull/1438)) [@harrism](https://github.com/harrism)
- Make device_memory_resource::supports_streams() not pure virtual. Remove derived implementations and calls in RMM ([#1437](https://github.com/rapidsai/rmm/pull/1437)) [@harrism](https://github.com/harrism)
- Deprecate rmm::mr::device_memory_resource::get_mem_info() and supports_get_mem_info(). ([#1436](https://github.com/rapidsai/rmm/pull/1436)) [@harrism](https://github.com/harrism)
- Support CUDA 12.2 ([#1419](https://github.com/rapidsai/rmm/pull/1419)) [@jameslamb](https://github.com/jameslamb)

## 🛠️ Improvements

- Use `conda env create --yes` instead of `--force` ([#1509](https://github.com/rapidsai/rmm/pull/1509)) [@bdice](https://github.com/bdice)
- Add upper bound to prevent usage of NumPy 2 ([#1501](https://github.com/rapidsai/rmm/pull/1501)) [@bdice](https://github.com/bdice)
- Remove hard-coding of RAPIDS version where possible ([#1496](https://github.com/rapidsai/rmm/pull/1496)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Require NumPy 1.23+ ([#1488](https://github.com/rapidsai/rmm/pull/1488)) [@jakirkham](https://github.com/jakirkham)
- Use `rmm::device_async_resource_ref` in multi_stream_allocation benchmark ([#1482](https://github.com/rapidsai/rmm/pull/1482)) [@miscco](https://github.com/miscco)
- Update devcontainers to CUDA Toolkit 12.2 ([#1470](https://github.com/rapidsai/rmm/pull/1470)) [@trxcllnt](https://github.com/trxcllnt)
- Add support for Python 3.11 ([#1469](https://github.com/rapidsai/rmm/pull/1469)) [@jameslamb](https://github.com/jameslamb)
- target branch-24.04 for GitHub Actions workflows ([#1468](https://github.com/rapidsai/rmm/pull/1468)) [@jameslamb](https://github.com/jameslamb)
- [FEA]: Use `std::optional` instead of `thrust::optional` ([#1464](https://github.com/rapidsai/rmm/pull/1464)) [@miscco](https://github.com/miscco)
- Add environment-agnostic scripts for running ctests and pytests ([#1462](https://github.com/rapidsai/rmm/pull/1462)) [@trxcllnt](https://github.com/trxcllnt)
- Ensure that `ctest` is called with `--no-tests=error`. ([#1460](https://github.com/rapidsai/rmm/pull/1460)) [@bdice](https://github.com/bdice)
- Update ops-bot.yaml ([#1458](https://github.com/rapidsai/rmm/pull/1458)) [@AyodeAwe](https://github.com/AyodeAwe)
- Adopt the `rmm::device_async_resource_ref` alias ([#1454](https://github.com/rapidsai/rmm/pull/1454)) [@miscco](https://github.com/miscco)
- Refactor error.hpp out of detail ([#1439](https://github.com/rapidsai/rmm/pull/1439)) [@lamarrr](https://github.com/lamarrr)

# RMM 24.02.00 (12 Feb 2024)

## 🚨 Breaking Changes

- Make device_memory_resource::do_get_mem_info() and supports_get_mem_info() not pure virtual. Remove derived implementations and calls in RMM ([#1430](https://github.com/rapidsai/rmm/pull/1430)) [@harrism](https://github.com/harrism)
- Deprecate detail::available_device_memory, most detail/aligned.hpp utilities, and optional pool_memory_resource initial size ([#1424](https://github.com/rapidsai/rmm/pull/1424)) [@harrism](https://github.com/harrism)
- Require explicit pool size in `pool_memory_resource` and move some things out of detail namespace ([#1417](https://github.com/rapidsai/rmm/pull/1417)) [@harrism](https://github.com/harrism)
- Remove HTML builds of librmm ([#1415](https://github.com/rapidsai/rmm/pull/1415)) [@vyasr](https://github.com/vyasr)
- Update to CCCL 2.2.0. ([#1404](https://github.com/rapidsai/rmm/pull/1404)) [@bdice](https://github.com/bdice)
- Switch to scikit-build-core ([#1287](https://github.com/rapidsai/rmm/pull/1287)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- Exclude tests from builds ([#1459](https://github.com/rapidsai/rmm/pull/1459)) [@vyasr](https://github.com/vyasr)
- Update CODEOWNERS ([#1410](https://github.com/rapidsai/rmm/pull/1410)) [@raydouglass](https://github.com/raydouglass)
- Correct signatures for torch allocator plug in ([#1407](https://github.com/rapidsai/rmm/pull/1407)) [@wence-](https://github.com/wence-)
- Fix Arena MR to support simultaneous access by PTDS and other streams ([#1395](https://github.com/rapidsai/rmm/pull/1395)) [@tgravescs](https://github.com/tgravescs)
- Fix else-after-throw clang tidy error ([#1391](https://github.com/rapidsai/rmm/pull/1391)) [@harrism](https://github.com/harrism)

## 📖 Documentation

- remove references to setup.py in docs ([#1420](https://github.com/rapidsai/rmm/pull/1420)) [@jameslamb](https://github.com/jameslamb)
- Remove HTML builds of librmm ([#1415](https://github.com/rapidsai/rmm/pull/1415)) [@vyasr](https://github.com/vyasr)
- Update GPU support docs to drop Pascal ([#1413](https://github.com/rapidsai/rmm/pull/1413)) [@harrism](https://github.com/harrism)

## 🚀 New Features

- Make device_memory_resource::do_get_mem_info() and supports_get_mem_info() not pure virtual. Remove derived implementations and calls in RMM ([#1430](https://github.com/rapidsai/rmm/pull/1430)) [@harrism](https://github.com/harrism)
- Deprecate detail::available_device_memory, most detail/aligned.hpp utilities, and optional pool_memory_resource initial size ([#1424](https://github.com/rapidsai/rmm/pull/1424)) [@harrism](https://github.com/harrism)
- Add a host-pinned memory resource that can be used as upstream for `pool_memory_resource`. ([#1392](https://github.com/rapidsai/rmm/pull/1392)) [@harrism](https://github.com/harrism)

## 🛠️ Improvements

- Remove usages of rapids-env-update ([#1423](https://github.com/rapidsai/rmm/pull/1423)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Refactor CUDA versions in dependencies.yaml. ([#1422](https://github.com/rapidsai/rmm/pull/1422)) [@bdice](https://github.com/bdice)
- Require explicit pool size in `pool_memory_resource` and move some things out of detail namespace ([#1417](https://github.com/rapidsai/rmm/pull/1417)) [@harrism](https://github.com/harrism)
- Update dependencies.yaml to support CUDA 12.*. ([#1414](https://github.com/rapidsai/rmm/pull/1414)) [@bdice](https://github.com/bdice)
- Define python dependency range as a matrix fallback. ([#1409](https://github.com/rapidsai/rmm/pull/1409)) [@bdice](https://github.com/bdice)
- Use latest cuda-python within CUDA major version. ([#1406](https://github.com/rapidsai/rmm/pull/1406)) [@bdice](https://github.com/bdice)
- Update to CCCL 2.2.0. ([#1404](https://github.com/rapidsai/rmm/pull/1404)) [@bdice](https://github.com/bdice)
- Remove RMM_BUILD_WHEELS and standardize Python builds ([#1401](https://github.com/rapidsai/rmm/pull/1401)) [@vyasr](https://github.com/vyasr)
- Update to fmt 10.1.1 and spdlog 1.12.0. ([#1374](https://github.com/rapidsai/rmm/pull/1374)) [@bdice](https://github.com/bdice)
- Switch to scikit-build-core ([#1287](https://github.com/rapidsai/rmm/pull/1287)) [@vyasr](https://github.com/vyasr)

# RMM 23.12.00 (6 Dec 2023)

## 🚨 Breaking Changes

- Document minimum CUDA version of 11.4 ([#1385](https://github.com/rapidsai/rmm/pull/1385)) [@harrism](https://github.com/harrism)
- Store and set the correct CUDA device in device_buffer ([#1370](https://github.com/rapidsai/rmm/pull/1370)) [@harrism](https://github.com/harrism)
- Use `cuda::mr::memory_resource` instead of raw `device_memory_resource` ([#1095](https://github.com/rapidsai/rmm/pull/1095)) [@miscco](https://github.com/miscco)

## 🐛 Bug Fixes

- Update actions/labeler to v4 ([#1397](https://github.com/rapidsai/rmm/pull/1397)) [@raydouglass](https://github.com/raydouglass)
- Backport arena MR fix for simultaneous access by PTDS and other streams ([#1396](https://github.com/rapidsai/rmm/pull/1396)) [@bdice](https://github.com/bdice)
- Deliberately leak PTDS thread_local events in stream ordered mr ([#1375](https://github.com/rapidsai/rmm/pull/1375)) [@wence-](https://github.com/wence-)
- Add missing CUDA 12 dependencies and fix dlopen library names ([#1366](https://github.com/rapidsai/rmm/pull/1366)) [@vyasr](https://github.com/vyasr)

## 📖 Documentation

- Document minimum CUDA version of 11.4 ([#1385](https://github.com/rapidsai/rmm/pull/1385)) [@harrism](https://github.com/harrism)
- Fix more doxygen issues ([#1367](https://github.com/rapidsai/rmm/pull/1367)) [@vyasr](https://github.com/vyasr)
- Add groups to the doxygen docs ([#1358](https://github.com/rapidsai/rmm/pull/1358)) [@vyasr](https://github.com/vyasr)
- Enable doxygen XML and fix issues ([#1348](https://github.com/rapidsai/rmm/pull/1348)) [@vyasr](https://github.com/vyasr)

## 🚀 New Features

- Make internally stored default argument values public ([#1373](https://github.com/rapidsai/rmm/pull/1373)) [@vyasr](https://github.com/vyasr)
- Store and set the correct CUDA device in device_buffer ([#1370](https://github.com/rapidsai/rmm/pull/1370)) [@harrism](https://github.com/harrism)
- Update rapids-cmake functions to non-deprecated signatures ([#1357](https://github.com/rapidsai/rmm/pull/1357)) [@robertmaynard](https://github.com/robertmaynard)
- Generate unified Python/C++ docs ([#1324](https://github.com/rapidsai/rmm/pull/1324)) [@vyasr](https://github.com/vyasr)
- Use `cuda::mr::memory_resource` instead of raw `device_memory_resource` ([#1095](https://github.com/rapidsai/rmm/pull/1095)) [@miscco](https://github.com/miscco)

## 🛠️ Improvements

- Silence false gcc warning ([#1381](https://github.com/rapidsai/rmm/pull/1381)) [@miscco](https://github.com/miscco)
- Build concurrency for nightly and merge triggers ([#1380](https://github.com/rapidsai/rmm/pull/1380)) [@bdice](https://github.com/bdice)
- Update `shared-action-workflows` references ([#1363](https://github.com/rapidsai/rmm/pull/1363)) [@AyodeAwe](https://github.com/AyodeAwe)
- Use branch-23.12 workflows. ([#1360](https://github.com/rapidsai/rmm/pull/1360)) [@bdice](https://github.com/bdice)
- Update devcontainers to 23.12 ([#1355](https://github.com/rapidsai/rmm/pull/1355)) [@raydouglass](https://github.com/raydouglass)
- Generate proper, consistent nightly versions for pip and conda packages ([#1347](https://github.com/rapidsai/rmm/pull/1347)) [@vyasr](https://github.com/vyasr)
- RMM: Build CUDA 12.0 ARM conda packages. ([#1330](https://github.com/rapidsai/rmm/pull/1330)) [@bdice](https://github.com/bdice)

# RMM 23.10.00 (11 Oct 2023)

## 🚨 Breaking Changes

- Update to Cython 3.0.0 ([#1313](https://github.com/rapidsai/rmm/pull/1313)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- Compile cdef public functions from torch_allocator with C ABI ([#1350](https://github.com/rapidsai/rmm/pull/1350)) [@wence-](https://github.com/wence-)
- Make doxygen only a conda dependency. ([#1344](https://github.com/rapidsai/rmm/pull/1344)) [@bdice](https://github.com/bdice)
- Use `conda mambabuild` not `mamba mambabuild` ([#1338](https://github.com/rapidsai/rmm/pull/1338)) [@wence-](https://github.com/wence-)
- Fix stream_ordered_memory_resource attempt to record event in stream from another device ([#1333](https://github.com/rapidsai/rmm/pull/1333)) [@harrism](https://github.com/harrism)

## 📖 Documentation

- Clean up headers in CMakeLists.txt. ([#1341](https://github.com/rapidsai/rmm/pull/1341)) [@bdice](https://github.com/bdice)
- Add pre-commit hook to validate doxygen ([#1334](https://github.com/rapidsai/rmm/pull/1334)) [@vyasr](https://github.com/vyasr)
- Fix doxygen warnings ([#1317](https://github.com/rapidsai/rmm/pull/1317)) [@vyasr](https://github.com/vyasr)
- Treat warnings as errors in Python documentation ([#1316](https://github.com/rapidsai/rmm/pull/1316)) [@vyasr](https://github.com/vyasr)

## 🚀 New Features

- Enable RMM Debug Logging via Python ([#1339](https://github.com/rapidsai/rmm/pull/1339)) [@harrism](https://github.com/harrism)

## 🛠️ Improvements

- Update image names ([#1346](https://github.com/rapidsai/rmm/pull/1346)) [@AyodeAwe](https://github.com/AyodeAwe)
- Update to clang 16.0.6. ([#1343](https://github.com/rapidsai/rmm/pull/1343)) [@bdice](https://github.com/bdice)
- Update doxygen to 1.9.1 ([#1337](https://github.com/rapidsai/rmm/pull/1337)) [@vyasr](https://github.com/vyasr)
- Simplify wheel build scripts and allow alphas of RAPIDS dependencies ([#1335](https://github.com/rapidsai/rmm/pull/1335)) [@divyegala](https://github.com/divyegala)
- Use `copy-pr-bot` ([#1329](https://github.com/rapidsai/rmm/pull/1329)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add RMM devcontainers ([#1328](https://github.com/rapidsai/rmm/pull/1328)) [@trxcllnt](https://github.com/trxcllnt)
- Add Python bindings for `limiting_resource_adaptor` ([#1327](https://github.com/rapidsai/rmm/pull/1327)) [@pentschev](https://github.com/pentschev)
- Fix missing jQuery error in docs ([#1321](https://github.com/rapidsai/rmm/pull/1321)) [@AyodeAwe](https://github.com/AyodeAwe)
- Use fetch_rapids.cmake. ([#1319](https://github.com/rapidsai/rmm/pull/1319)) [@bdice](https://github.com/bdice)
- Update to Cython 3.0.0 ([#1313](https://github.com/rapidsai/rmm/pull/1313)) [@vyasr](https://github.com/vyasr)
- Branch 23.10 merge 23.08 ([#1312](https://github.com/rapidsai/rmm/pull/1312)) [@vyasr](https://github.com/vyasr)
- Branch 23.10 merge 23.08 ([#1309](https://github.com/rapidsai/rmm/pull/1309)) [@vyasr](https://github.com/vyasr)

# RMM 23.08.00 (9 Aug 2023)

## 🚨 Breaking Changes

- Stop invoking setup.py ([#1300](https://github.com/rapidsai/rmm/pull/1300)) [@vyasr](https://github.com/vyasr)
- Remove now-deprecated top-level allocator functions ([#1281](https://github.com/rapidsai/rmm/pull/1281)) [@wence-](https://github.com/wence-)
- Remove padding from device_memory_resource ([#1278](https://github.com/rapidsai/rmm/pull/1278)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- Fix typo in wheels-test.yaml. ([#1310](https://github.com/rapidsai/rmm/pull/1310)) [@bdice](https://github.com/bdice)
- Add a missing &#39;#include &lt;array&gt;&#39; in logger.hpp ([#1295](https://github.com/rapidsai/rmm/pull/1295)) [@valgur](https://github.com/valgur)
- Use gbench `thread_index()` accessor to fix replay bench compilation ([#1293](https://github.com/rapidsai/rmm/pull/1293)) [@harrism](https://github.com/harrism)
- Ensure logger tests don&#39;t generate temp directories in build dir ([#1289](https://github.com/rapidsai/rmm/pull/1289)) [@robertmaynard](https://github.com/robertmaynard)

## 🚀 New Features

- Remove now-deprecated top-level allocator functions ([#1281](https://github.com/rapidsai/rmm/pull/1281)) [@wence-](https://github.com/wence-)

## 🛠️ Improvements

- Switch to new CI wheel building pipeline ([#1305](https://github.com/rapidsai/rmm/pull/1305)) [@vyasr](https://github.com/vyasr)
- Revert CUDA 12.0 CI workflows to branch-23.08. ([#1303](https://github.com/rapidsai/rmm/pull/1303)) [@bdice](https://github.com/bdice)
- Update linters: remove flake8, add ruff, update cython-lint ([#1302](https://github.com/rapidsai/rmm/pull/1302)) [@vyasr](https://github.com/vyasr)
- Adding identify minimum version requirement ([#1301](https://github.com/rapidsai/rmm/pull/1301)) [@hyperbolic2346](https://github.com/hyperbolic2346)
- Stop invoking setup.py ([#1300](https://github.com/rapidsai/rmm/pull/1300)) [@vyasr](https://github.com/vyasr)
- Use cuda-version to constrain cudatoolkit. ([#1296](https://github.com/rapidsai/rmm/pull/1296)) [@bdice](https://github.com/bdice)
- Update to CMake 3.26.4 ([#1291](https://github.com/rapidsai/rmm/pull/1291)) [@vyasr](https://github.com/vyasr)
- use rapids-upload-docs script ([#1288](https://github.com/rapidsai/rmm/pull/1288)) [@AyodeAwe](https://github.com/AyodeAwe)
- Reorder parameters in RMM_EXPECTS ([#1286](https://github.com/rapidsai/rmm/pull/1286)) [@vyasr](https://github.com/vyasr)
- Remove documentation build scripts for Jenkins ([#1285](https://github.com/rapidsai/rmm/pull/1285)) [@ajschmidt8](https://github.com/ajschmidt8)
- Remove padding from device_memory_resource ([#1278](https://github.com/rapidsai/rmm/pull/1278)) [@vyasr](https://github.com/vyasr)
- Unpin scikit-build upper bound ([#1275](https://github.com/rapidsai/rmm/pull/1275)) [@vyasr](https://github.com/vyasr)
- RMM: Build CUDA 12 packages ([#1223](https://github.com/rapidsai/rmm/pull/1223)) [@bdice](https://github.com/bdice)

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
