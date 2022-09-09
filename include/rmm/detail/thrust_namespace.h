/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#pragma once

#include <thrust/detail/config.h>  // namespace macros

#ifdef THRUST_WRAPPED_NAMESPACE

// Ensure the namespace exist before we import it
// so that this include can occur before thrust includes
namespace THRUST_WRAPPED_NAMESPACE {
namespace thrust {
}
}  // namespace THRUST_WRAPPED_NAMESPACE

namespace rmm {
using namespace THRUST_WRAPPED_NAMESPACE;
}

#endif
