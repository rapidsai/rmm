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
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/fixed_multisize_memory_resource.hpp>
#include <rmm/mr/device/hybrid_memory_resource.hpp>
#include <rmm/mr/device/thread_safe_resource_adaptor.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>

#define RMM_HYBRID_RESOURCE

#ifdef RMM_HYBRID_RESOURCE
  // nested MR type names can get long...
  using pool_mr = rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>;
  using mng_pool_mr = rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource>;
  using fixed_multisize_mr = rmm::mr::fixed_multisize_memory_resource<pool_mr>;
  using mng_fixed_multisize_mr = rmm::mr::fixed_multisize_memory_resource<mng_pool_mr>;
  using hybrid_mr = rmm::mr::hybrid_memory_resource<fixed_multisize_mr, pool_mr>;
  using mng_hybrid_mr = rmm::mr::hybrid_memory_resource<mng_fixed_multisize_mr, mng_pool_mr>;
  using safe_hybrid_mr = rmm::mr::thread_safe_resource_adaptor<hybrid_mr>;
  using safe_mng_hybrid_mr = rmm::mr::thread_safe_resource_adaptor<mng_hybrid_mr>;
  using logging_safe_hybrid_mr = rmm::mr::logging_resource_adaptor<safe_hybrid_mr>;
  using logging_safe_mng_hybrid_mr = rmm::mr::logging_resource_adaptor<safe_mng_hybrid_mr>;
#endif

namespace rmm {
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

// Initialize the manager
void Manager::initialize(const rmmOptions_t* new_options) {
  std::lock_guard<std::mutex> guard(manager_mutex);

  // repeat initialization is a no-op
  if (isInitialized()) return;

  if (nullptr != new_options) options = *new_options;

  if (usePoolAllocator()) {
#ifdef RMM_HYBRID_RESOURCE
    auto pool_size = getOptions().initial_pool_size;
    auto const& devices = getOptions().devices;
    if (useManagedMemory()) {
      mng_pool_mr* pool = new mng_pool_mr(new rmm::mr::managed_memory_resource, pool_size);
      auto hybrid = new safe_mng_hybrid_mr(
          new mng_hybrid_mr(new mng_fixed_multisize_mr(pool), pool));
    
      if (getOptions().enable_logging)
        initialized_resource.reset(new logging_safe_mng_hybrid_mr(hybrid, "rmm_log.csv"));
      else
        initialized_resource.reset(hybrid);
    } else {
      pool_mr* pool = new pool_mr(new rmm::mr::cuda_memory_resource, pool_size);
      auto hybrid = new safe_hybrid_mr(new hybrid_mr(new fixed_multisize_mr(pool), pool));

      if (getOptions().enable_logging)
        initialized_resource.reset(new logging_safe_hybrid_mr(hybrid, "rmm_log.csv"));
      else
        initialized_resource.reset(hybrid);
    }
#else
    auto pool_size = getOptions().initial_pool_size;
    auto const& devices = getOptions().devices;
    if (useManagedMemory()) {
      initialized_resource.reset(
        new rmm::mr::cnmem_managed_memory_resource(pool_size, devices));
    } else {
      initialized_resource.reset(
        new rmm::mr::cnmem_memory_resource(pool_size,  devices));
    }
#endif
  } else if (rmm::Manager::useManagedMemory()) {
    initialized_resource.reset(new rmm::mr::managed_memory_resource());
  } else {
    initialized_resource.reset(new rmm::mr::cuda_memory_resource());
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
#ifdef RMM_HYBRID_RESOURCE
    // TODO: it's a pain to free all these
    if (usePoolAllocator()) {
      if (useManagedMemory()) {
        safe_mng_hybrid_mr *ts{
          (getOptions().enable_logging) ? 
            static_cast<logging_safe_mng_hybrid_mr*>(initialized_resource.get())->get_upstream() :
            static_cast<safe_mng_hybrid_mr*>(initialized_resource.get())
        };
        auto hybrid = ts->get_upstream();
        delete hybrid->get_small_mr();
        delete hybrid->get_large_mr();
        delete hybrid;
      } else {
        safe_hybrid_mr *ts{
          (getOptions().enable_logging) ? 
          static_cast<logging_safe_hybrid_mr*>(initialized_resource.get())->get_upstream() :
          static_cast<safe_hybrid_mr*>(initialized_resource.get())
        };
        auto hybrid = ts->get_upstream();
        delete hybrid->get_small_mr();
        delete hybrid->get_large_mr();
        delete hybrid;
      }
    }
#endif
    initialized_resource.reset();
    is_initialized = false;
  }
}
}  // namespace rmm
