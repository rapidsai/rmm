/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

// Check if LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE is defined
#ifndef LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
#error \
  "RMM requires LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE to be defined. This is typically done by the build system. If you're seeing this error, you may be using a custom build setup that doesn't define this flag. Please add -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE to your compiler flags."
#endif

#include <cuda/memory_resource>
