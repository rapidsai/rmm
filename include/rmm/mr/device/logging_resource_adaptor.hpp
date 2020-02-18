/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "device_memory_resource.hpp"

#include <rmm/detail/error.hpp>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

namespace rmm {
namespace mr {
/**
 * @brief Resource that uses `Upstream` to allocate memory and logs information
 * about the requested allocation/deallocations.
 *
 * @tparam Upstream Type of the upstream resource used for
 * allocation/deallocation.
 */
template <typename Upstream>
class logging_resource_adaptor final : public device_memory_resource {
 public:
  /**
   * @brief Construct a new logging resource adaptor using `upstream` to satisfy
   * allocation requests and logging information about each allocation/free to
   * the file specified by `filename`.
   *
   * If the directories specified in `filename` do not exist, they will be
   * created.
   *
   * @throws `rmm::logic_error` if `upstream == nullptr`
   * @throws `spdlog::spdlog_ex` if opening `filename` failed
   *
   * @param upstream The resource used for allocating/deallocating device memory
   * @param filename Name of file to write log info
   */
  logging_resource_adaptor(Upstream* upstream, std::string const& filename)
      : upstream_{upstream}, logger_{spdlog::basic_logger_mt("RMM", filename)} {
    RMM_EXPECTS(nullptr != upstream,
                "Unexpected null upstream resource pointer.");
  }

 private:
  /**
   * @brief Allocates memory of size at least `bytes` using the upstream
   * resource and logs the allocation.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `rmm::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @param stream Stream on which to perform the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override {
    return upstream_->allocate(bytes, stream);
  }

  /**
   * @brief Free allocation of size `bytes` pointed to to by `p` and log the
   * deallocation.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   * @param bytes Size of the allocation
   * @param stream Stream on which to perform the deallocation
   */
  void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override {
    upstream_->deallocate(p, bytes, stream);
  }

  /**
   * @brief Compare the upstream resource to another.
   *
   * @throws Nothing.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  bool do_is_equal(device_memory_resource const& other) const noexcept {
    return upstream_->is_equal(other);
  }

  /**
   * @brief Get free and available memory from upstream resource.
   *
   * @throws `rmm::cuda_error` if unable to retrieve memory info.
   *
   * @param stream Stream on which to get the mem info.
   * @return std::pair contaiing free_size and total_size of memory
   */
  std::pair<size_t, size_t> do_get_mem_info(cudaStream_t stream) const {
    return upstream_->get_mem_info(stream);
  }

  std::shared_ptr<spdlog::logger> logger_;  ///< spdlog logger object

  Upstream* upstream_;  ///< The upstream resource used for satisfying
                        ///< allocation requests
};
}  // namespace mr
}  // namespace rmm