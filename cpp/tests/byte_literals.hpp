/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>

namespace rmm::test {

constexpr auto kilo{long{1} << 10};
constexpr auto mega{long{1} << 20};
constexpr auto giga{long{1} << 30};
constexpr auto tera{long{1} << 40};
constexpr auto peta{long{1} << 50};

// user-defined Byte literals
constexpr unsigned long long operator""_B(unsigned long long val) { return val; }
constexpr unsigned long long operator""_KiB(unsigned long long const val) { return kilo * val; }
constexpr unsigned long long operator""_MiB(unsigned long long const val) { return mega * val; }
constexpr unsigned long long operator""_GiB(unsigned long long const val) { return giga * val; }
constexpr unsigned long long operator""_TiB(unsigned long long const val) { return tera * val; }
constexpr unsigned long long operator""_PiB(unsigned long long const val) { return peta * val; }

}  // namespace rmm::test
