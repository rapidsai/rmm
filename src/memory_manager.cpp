/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <rmm/detail/memory_manager.hpp>
#include <rmm/mr/device/cnmem_managed_memory_resource.hpp>
#include <rmm/mr/device/cnmem_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>

namespace rmm {

using cuda_mr = rmm::mr::cuda_memory_resource;
using pool_mr = rmm::mr::cnmem_memory_resource;
using managed_mr = rmm::mr::cnmem_managed_memory_resource;
using pool_managed_mr = rmm::mr::cnmem_managed_memory_resource;
using logging_pool_mr = rmm::mr::logging_resource_adaptor<pool_mr>;
using logging_pool_managed_mr = rmm::mr::logging_resource_adaptor<pool_managed_mr>;

/**
 * Record a memory manager event in the log.
 *
 * @param[in] event The type of event (Alloc, Realloc, or Free)
 * @param[in] DeviceId The device to which this event applies.
 * @param[in] ptr The device pointer being allocated or freed.
 * @param[in] t The timestamp to record.
 * @param[in] size The size of allocation (only needed for Alloc/Realloc).
 * @param[in] stream The stream on which the allocation is happening
 *                   (only needed for Alloc/Realloc).
 */
void Logger::record(MemEvent_t event, int deviceId, void* ptr, TimePt start,
                    TimePt end, size_t freeMem, size_t totalMem, size_t size,
                    cudaStream_t stream, std::string filename,
                    unsigned int line)

{
  std::lock_guard<std::mutex> guard(log_mutex);
  if (Alloc == event)
    current_allocations.insert(ptr);
  else if (Free == event)
    current_allocations.erase(ptr);
  events.push_back({event, deviceId, ptr, size, stream, freeMem, totalMem,
                    current_allocations.size(), start, end, filename, line});
}

/**
 * @brief Output a comma-separated value string of the current log to the
 *        provided ostream
 *
 * @param[in] csv The output stream to put the CSV log string into.
 */
void Logger::to_csv(std::ostream& csv) {
  csv << "Event Type,Device ID,Address,Stream,Size (bytes),Free Memory,"
      << "Total Memory,Current Allocs,Start,End,Elapsed,Location\n";

  for (auto& e : events) {
    auto event_str = "Alloc";
    if (e.event == Realloc) event_str = "Realloc";
    if (e.event == Free) event_str = "Free";

    std::chrono::duration<double> elapsed = e.end - e.start;

    csv << event_str << "," << e.deviceId << "," << e.ptr << "," << e.stream
        << "," << e.size << "," << e.freeMem << "," << e.totalMem << ","
        << e.currentAllocations << ","
        << std::chrono::duration<double>(e.start - base_time).count() << ","
        << std::chrono::duration<double>(e.end - base_time).count() << ","
        << elapsed.count() << "," << e.filename << ":" << e.line << std::endl;
  }
}

/**
 * @brief Clear the log
 */
void Logger::clear() {
  std::lock_guard<std::mutex> guard(log_mutex);
  events.clear();
}

rmmError_t Manager::registerStream(cudaStream_t stream) {
  std::lock_guard<std::mutex> guard(manager_mutex);
  if (registered_streams.empty() || 0 == registered_streams.count(stream)) {
    registered_streams.insert(stream);
    if (stream &&
        usePoolAllocator())  // don't register the null stream with CNMem
      RMM_CHECK_CNMEM(cnmemRegisterStream(stream));
  }
  return RMM_SUCCESS;
}

// reset the initialized resource, enabling logging if set in options
template <typename MemoryResource>
void reset_resource(std::unique_ptr<mr::device_memory_resource>& initialized_resource,
                    MemoryResource *mr,
                    bool enable_logging) {
  if (enable_logging) {
    initialized_resource.reset(new rmm::mr::logging_resource_adaptor<MemoryResource>(mr));
  } else {
    initialized_resource.reset(mr);
  }
}

// Initialize the manager
void Manager::initialize(const rmmOptions_t* new_options) {
  std::lock_guard<std::mutex> guard(manager_mutex);

  // repeat initialization is a no-op
  if (isInitialized()) return;

  if (nullptr != new_options) options = *new_options;

  bool enable_logging = getOptions().enable_logging;

  if (usePoolAllocator()) {
    auto pool_size = getOptions().initial_pool_size;
    auto const& devices = getOptions().devices;

    if (useManagedMemory()) {
      reset_resource(initialized_resource, new pool_managed_mr(pool_size, devices), enable_logging);
    } else {
      reset_resource(initialized_resource, new pool_mr(pool_size, devices), enable_logging);
    }
  } else if (rmm::Manager::useManagedMemory()) {
    reset_resource(initialized_resource, new managed_mr(), enable_logging);
  } else {
    reset_resource(initialized_resource, new cuda_mr(), enable_logging);
  }

  rmm::mr::set_default_resource(initialized_resource.get());

  is_initialized = true;
}

// Shut down the Manager (clears the context)
void Manager::finalize() {
  std::lock_guard<std::mutex> guard(manager_mutex);

  // finalization before initialization is a no-op
  if (isInitialized()) {
    registered_streams.clear();
    logger.clear();
    initialized_resource.reset();
    is_initialized = false;
  }
}
}  // namespace rmm
