/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <mutex>
#include <set>

#include <rmm/mr/device/detail/coalescing_free_list.hpp>
#include <rmm/mr/device/detail/free_list.hpp>

namespace rmm {
namespace mr {
namespace detail {

struct arena {
  using free_list           = coalescing_free_list;
  using block_type          = free_list::block_type;
  using split_block_type    = split_block<block_type>;
  using compare_blocks_type = compare_blocks<block_type>;
  using allocated_set       = std::set<block_type, compare_blocks_type>;

  free_list free_blocks;
  allocated_set allocated_blocks;
  mutable std::mutex mtx;
};

}  // namespace detail
}  // namespace mr
}  // namespace rmm
