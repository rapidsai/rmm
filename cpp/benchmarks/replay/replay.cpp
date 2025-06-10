/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either ex  ess or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/logger.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

#include <benchmark/benchmark.h>
#include <benchmarks/utilities/cxxopts.hpp>
#include <benchmarks/utilities/log_parser.hpp>
#include <benchmarks/utilities/simulated_memory_resource.hpp>

#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <thread>

/// MR factory functions
std::shared_ptr<rmm::mr::device_memory_resource> make_cuda(std::size_t = 0)
{
  return std::make_shared<rmm::mr::cuda_memory_resource>();
}

std::shared_ptr<rmm::mr::device_memory_resource> make_managed(std::size_t = 0)
{
  return std::make_shared<rmm::mr::managed_memory_resource>();
}

std::shared_ptr<rmm::mr::device_memory_resource> make_simulated(std::size_t simulated_size)
{
  return std::make_shared<rmm::mr::simulated_memory_resource>(simulated_size);
}

inline auto make_pool(std::size_t simulated_size)
{
  if (simulated_size > 0) {
    return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      make_simulated(simulated_size), simulated_size, simulated_size);
  }
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_cuda(), 0);
}

inline auto make_arena(std::size_t simulated_size)
{
  if (simulated_size > 0) {
    return rmm::mr::make_owning_wrapper<rmm::mr::arena_memory_resource>(
      make_simulated(simulated_size), simulated_size, simulated_size);
  }
  return rmm::mr::make_owning_wrapper<rmm::mr::arena_memory_resource>(make_cuda());
}

inline auto make_binning(std::size_t simulated_size)
{
  auto pool = make_pool(simulated_size);
  auto mr   = rmm::mr::make_owning_wrapper<rmm::mr::binning_memory_resource>(pool);
  const auto min_size_exp{18};
  const auto max_size_exp{22};
  for (std::size_t i = min_size_exp; i <= max_size_exp; i++) {
    mr->wrapped().add_bin(1 << i);
  }
  return mr;
}

using MRFactoryFunc = std::function<std::shared_ptr<rmm::mr::device_memory_resource>(std::size_t)>;

/**
 * @brief Represents an allocation made during the replay
 *
 */
struct allocation {
  allocation() = default;
  void* ptr{};
  allocation(void* ptr, std::size_t size) : ptr{ptr}, size{size} {}
  std::size_t size{};
};

/**
 * @brief Function object for running a replay benchmark with the specified
 * `device_memory_resource`.
 *
 * @tparam MR The type of the `device_memory_resource` to use for allocation
 * replay
 */
struct replay_benchmark {
  MRFactoryFunc factory_;
  std::size_t simulated_size_;
  std::shared_ptr<rmm::mr::device_memory_resource> mr_{};
  std::vector<std::vector<rmm::detail::event>> const& events_{};

  // Maps a pointer from the event log to an active allocation
  std::unordered_map<uintptr_t, allocation> allocation_map;

  std::condition_variable cv;  // to ensure in-order playback
  std::mutex event_mutex;      // to make event_index and allocation_map thread-safe
  std::size_t event_index{0};  // playback index
  std::barrier<> barrier_;     // barrier to sequence resetting of event_index

  /**
   * @brief Construct a `replay_benchmark` from a list of events and
   * set of arguments forwarded to the MR constructor.
   *
   * @param factory A factory function to create the memory resource
   * @param events The set of allocation events to replay
   * @param args Variable number of arguments forward to the constructor of MR
   */
  replay_benchmark(MRFactoryFunc factory,
                   std::size_t simulated_size,
                   std::vector<std::vector<rmm::detail::event>> const& events)
    : factory_{std::move(factory)},
      simulated_size_{simulated_size},
      events_{events},
      allocation_map{events.size()},
      barrier_{static_cast<std::ptrdiff_t>(events_.size())}
  {
  }

  /**
   * @brief Move construct a replay_benchmark (needed by RegisterBenchmark)
   *
   * Does not copy the mutex or the map
   */
  replay_benchmark(replay_benchmark&& other) noexcept
    : factory_{std::move(other.factory_)},
      simulated_size_{other.simulated_size_},
      mr_{std::move(other.mr_)},
      events_{other.events_},
      allocation_map{std::move(other.allocation_map)},
      barrier_{static_cast<std::ptrdiff_t>(events_.size())}
  {
  }

  ~replay_benchmark()                                            = default;
  replay_benchmark(replay_benchmark const&)                      = delete;
  replay_benchmark& operator=(replay_benchmark const&)           = delete;
  replay_benchmark& operator=(replay_benchmark&& other) noexcept = delete;

  /// Add an allocation to the map (NOT thread safe)
  void set_allocation(uintptr_t ptr, allocation alloc) { allocation_map.insert({ptr, alloc}); }

  /// Remove an allocation from the map (NOT thread safe)
  allocation remove_allocation(uintptr_t ptr)
  {
    auto iter = allocation_map.find(ptr);
    if (iter != allocation_map.end()) {
      allocation alloc = iter->second;
      allocation_map.erase(iter);
      return alloc;
    }
    return allocation{};
  }

  /// Create the memory resource shared by all threads before the benchmark runs
  void SetUp(const ::benchmark::State& state)
  {
    if (state.thread_index() == 0) {
      RMM_LOG_INFO("------ Start of Benchmark -----");
      mr_ = factory_(simulated_size_);
    }
    // Can't release threads until MR is set up.
    barrier_.arrive_and_wait();
  }

  /// Destroy the memory resource and count any unallocated memory
  void TearDown(const ::benchmark::State& state)
  {
    // Can't tear down the MR until every thread is done.
    barrier_.arrive_and_wait();
    if (state.thread_index() == 0) {
      RMM_LOG_INFO("------ End of Benchmark -----");
      // clean up any leaked allocations
      std::size_t total_leaked{0};
      std::size_t num_leaked{0};
      for (auto const& ptr_alloc : allocation_map) {
        auto alloc = ptr_alloc.second;
        num_leaked++;
        total_leaked += alloc.size;
        mr_->deallocate(alloc.ptr, alloc.size);
      }
      if (num_leaked > 0) {
        std::cout << "LOG shows leak of " << num_leaked << " allocations of " << total_leaked
                  << " total bytes\n";
      }
      allocation_map.clear();
      mr_.reset();
    }
  }

  /// Run the replay benchmark
  void operator()(::benchmark::State& state)
  {
    SetUp(state);

    auto const& my_events = events_.at(state.thread_index());
    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores)
      // At start of each iteration event_index must be reset.
      // Any thread could do this, but this is easy
      if (state.thread_index() == 0) { event_index = 0; }
      // And everyone waits for the reset.
      barrier_.arrive_and_wait();
      std::for_each(my_events.begin(), my_events.end(), [this](auto event) {
        // ensure correct ordering between threads
        std::unique_lock<std::mutex> lock{event_mutex};
        if (event_index != event.index) {
          cv.wait(lock, [&]() { return event_index == event.index; });
        }

        // rmm::detail::action::ALLOCATE_FAILURE is ignored.
        if (rmm::detail::action::ALLOCATE == event.act) {
          auto ptr = mr_->allocate(event.size);
          set_allocation(event.pointer, allocation{ptr, event.size});
        } else if (rmm::detail::action::FREE == event.act) {
          auto alloc = remove_allocation(event.pointer);
          mr_->deallocate(alloc.ptr, event.size);
        }

        event_index++;
        cv.notify_all();
      });
      // Everyone waits to be done (so that the reset of the next
      // iteration doesn't proceed until we're finished)
      barrier_.arrive_and_wait();
    }

    TearDown(state);
  }
};

/**
 * @brief Processes a log file into a set of per-thread vectors of events
 *
 * @param filename Name of log file
 * @return A vector of events for each thread in the log
 */
std::vector<std::vector<rmm::detail::event>> parse_per_thread_events(std::string const& filename)
{
  using rmm::detail::event;
  std::vector<event> all_events = rmm::detail::parse_csv(filename);

  RMM_EXPECTS(std::all_of(all_events.begin(),
                          all_events.end(),
                          [](auto const& event) {
                            cudaStream_t custream;
                            memcpy(&custream, &event.stream, sizeof(cudaStream_t));
                            auto stream = rmm::cuda_stream_view{custream};
                            return stream.is_default() or stream.is_per_thread_default();
                          }),
              "Non-default streams not currently supported.");

  // Sort events by thread id
  std::stable_sort(all_events.begin(), all_events.end(), [](auto lhs, auto rhs) {
    return lhs.thread_id < rhs.thread_id;
  });

  // Count the number of events per thread
  std::vector<std::size_t> events_per_thread{};
  thrust::reduce_by_key(
    thrust::host,
    all_events.begin(),
    all_events.end(),
    thrust::make_constant_iterator(1),
    thrust::make_discard_iterator(),
    std::back_inserter(events_per_thread),
    [](event const& lhs, event const& rhs) { return lhs.thread_id == rhs.thread_id; });

  auto const num_threads = events_per_thread.size();

  // Copy each thread's events into its own vector
  std::vector<std::vector<event>> per_thread_events(num_threads);
  std::transform(events_per_thread.begin(),
                 events_per_thread.end(),
                 per_thread_events.begin(),
                 [&all_events, offset = 0](auto num_events) mutable {
                   auto begin = offset;
                   offset += num_events;
                   auto end = offset;
                   std::vector<event> thread_events(all_events.cbegin() + begin,
                                                    all_events.cbegin() + end);
                   // sort into original order
                   std::sort(thread_events.begin(), thread_events.end(), [](auto lhs, auto rhs) {
                     return lhs.index < rhs.index;
                   });
                   return thread_events;
                 });

  return per_thread_events;
}

void declare_benchmark(std::string const& name,
                       std::size_t simulated_size,
                       std::vector<std::vector<rmm::detail::event>> const& per_thread_events,
                       std::size_t num_threads)
{
  if (name == "cuda") {
    benchmark::RegisterBenchmark("CUDA Resource",
                                 replay_benchmark(&make_cuda, simulated_size, per_thread_events))
      ->Unit(benchmark::kMillisecond)
      ->Threads(static_cast<int>(num_threads));
  } else if (name == "binning") {
    benchmark::RegisterBenchmark("Binning Resource",
                                 replay_benchmark(&make_binning, simulated_size, per_thread_events))
      ->Unit(benchmark::kMillisecond)
      ->Threads(static_cast<int>(num_threads));
  } else if (name == "pool") {
    benchmark::RegisterBenchmark("Pool Resource",
                                 replay_benchmark(&make_pool, simulated_size, per_thread_events))
      ->Unit(benchmark::kMillisecond)
      ->Threads(static_cast<int>(num_threads));
  } else if (name == "arena") {
    benchmark::RegisterBenchmark("Arena Resource",
                                 replay_benchmark(&make_arena, simulated_size, per_thread_events))
      ->Unit(benchmark::kMillisecond)
      ->Threads(static_cast<int>(num_threads));
  } else if (name == "managed") {
    benchmark::RegisterBenchmark("Managed Resource",
                                 replay_benchmark(&make_managed, simulated_size, per_thread_events))
      ->Unit(benchmark::kMillisecond)
      ->Threads(static_cast<int>(num_threads));
  } else {
    std::cout << "Error: invalid memory_resource name: " << name << "\n";
  }
}

// Usage: REPLAY_BENCHMARK -f "path/to/log/file"
int main(int argc, char** argv)
{
  try {
    // benchmark::Initialize will remove GBench command line arguments it
    // recognizes and leave any remaining arguments
    ::benchmark::Initialize(&argc, argv);

    // Parse for replay arguments:
    auto args = [&argc, &argv]() {
      cxxopts::Options options(
        "RMM Replay Benchmark",
        "Replays and benchmarks allocation activity captured from RMM logging.");

      options.add_options()("f,file", "Name of RMM log file.", cxxopts::value<std::string>());
      options.add_options()("r,resource",
                            "Type of device_memory_resource",
                            cxxopts::value<std::string>()->default_value("pool"));
      options.add_options()(
        "s,size",
        "Size of simulated GPU memory in GiB. Not supported for the cuda memory "
        "resource.",
        cxxopts::value<float>()->default_value("0"));
      options.add_options()("v,verbose",
                            "Enable verbose printing of log events",
                            cxxopts::value<bool>()->default_value("false"));

      auto args = options.parse(argc, argv);

      if (args.count("file") == 0) {
        std::cout << options.help() << std::endl;
        exit(0);
      }

      return args;
    }();

    auto filename = args["file"].as<std::string>();

    auto per_thread_events = [filename]() {
      try {
        auto events = parse_per_thread_events(filename);
        return events;
      } catch (std::exception const& e) {
        std::cout << "Failed to parse events: " << e.what() << std::endl;
        return std::vector<std::vector<rmm::detail::event>>{};
      }
    }();

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    std::cout << "Using CUDA per-thread default stream.\n";
#endif

    auto const simulated_size =
      static_cast<std::size_t>(args["size"].as<float>() * static_cast<float>(1U << 30U));
    if (simulated_size != 0 && args["resource"].as<std::string>() != "cuda") {
      std::cout << "Simulating GPU with memory size of " << simulated_size << " bytes.\n";
    }

    std::cout << "Total Events: "
              << std::accumulate(
                   per_thread_events.begin(),
                   per_thread_events.end(),
                   0,
                   [](std::size_t accum, auto const& events) { return accum + events.size(); })
              << std::endl;

    for (std::size_t thread = 0; thread < per_thread_events.size(); ++thread) {
      std::cout << "Thread " << thread << ": " << per_thread_events[thread].size() << " events\n";
      if (args["verbose"].as<bool>()) {
        for (auto const& event : per_thread_events[thread]) {
          std::cout << event << std::endl;
        }
      }
    }

    auto const num_threads = per_thread_events.size();

    // Uncomment to enable / change default log level
    // rmm::logger().set_level(rapids_logger::level_enum::trace);

    if (args.count("resource") > 0) {
      std::string mr_name = args["resource"].as<std::string>();
      declare_benchmark(mr_name, simulated_size, per_thread_events, num_threads);
    } else {
      std::array<std::string, 5> mrs{"pool", "arena", "binning", "cuda", "managed"};
      std::for_each(std::cbegin(mrs),
                    std::cend(mrs),
                    [&simulated_size, &per_thread_events, &num_threads](auto const& mr) {
                      declare_benchmark(mr, simulated_size, per_thread_events, num_threads);
                    });
    }

    ::benchmark::RunSpecifiedBenchmarks();
  } catch (std::exception const& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  return 0;
}
