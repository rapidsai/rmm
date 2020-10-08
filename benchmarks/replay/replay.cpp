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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either ex  ess or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <atomic>
#include <benchmarks/utilities/cxxopts.hpp>
#include <benchmarks/utilities/log_parser.hpp>

#include <rmm/detail/error.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

#include <benchmark/benchmark.h>

#include <chrono>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <thread>

#include "spdlog/common.h"

/// MR factory functions
inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

inline auto make_pool()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_cuda());
}

inline auto make_arena()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::arena_memory_resource>(make_cuda());
}

inline auto make_binning()
{
  auto pool = make_pool();
  auto mr   = rmm::mr::make_owning_wrapper<rmm::mr::binning_memory_resource>(pool);
  for (std::size_t i = 18; i <= 22; i++) {
    mr->wrapped().add_bin(1 << i);
  }
  return mr;
}

using MRFactoryFunc = std::function<std::shared_ptr<rmm::mr::device_memory_resource>()>;

/**
 * @brief Represents an allocation made during the replay
 *
 */
struct allocation {
  allocation() = default;
  allocation(void* p_, std::size_t size_) : p{p_}, size{size_} {}
  void* p{};
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
  std::shared_ptr<rmm::mr::device_memory_resource> mr_{};
  std::vector<std::vector<rmm::detail::event>> const& events_{};

  // Maps a pointer from the event log to an active allocation
  std::unordered_map<uintptr_t, allocation> allocation_map;

  std::condition_variable cv;  // to ensure in-order playback
  std::mutex event_mutex;      // to make event_index and allocation_map thread-safe
  std::size_t event_index{0};  // playback index

  /**
   * @brief Construct a `replay_benchmark` from a list of events and
   * set of arguments forwarded to the MR constructor.
   *
   * @param factory A factory function to create the memory resource
   * @param events The set of allocation events to replay
   * @param args Variable number of arguments forward to the constructor of MR
   */
  replay_benchmark(MRFactoryFunc factory,
                   std::vector<std::vector<rmm::detail::event>> const& events)
    : factory_{factory}, mr_{}, events_{events}, allocation_map{events.size()}, event_index{0}
  {
  }

  /**
   * @brief Move construct a replay_benchmark (needed by RegisterBenchmark)
   *
   * Does not copy the mutex or the map
   */
  replay_benchmark(replay_benchmark&& other)
    : factory_{std::move(other.factory_)},
      mr_{std::move(other.mr_)},
      events_{std::move(other.events_)},
      allocation_map{events_.size()},
      event_index{0}
  {
  }

  replay_benchmark(replay_benchmark const&) = delete;

  /// Add an allocation to the map (NOT thread safe)
  void set_allocation(std::unordered_map<uintptr_t, allocation>& allocation_map,
                      uintptr_t ptr,
                      allocation alloc)
  {
    allocation_map.insert({ptr, alloc});
  }

  /// Remove an allocation from the map (NOT thread safe)
  allocation remove_allocation(std::unordered_map<uintptr_t, allocation>& allocation_map,
                               uintptr_t ptr)
  {
    auto iter = allocation_map.find(ptr);
    if (iter != allocation_map.end()) {
      allocation a = iter->second;
      allocation_map.erase(iter);
      return a;
    }
    return allocation{};
  }

  /// Create the memory resource shared by all threads before the benchmark runs
  void SetUp(const ::benchmark::State& state)
  {
    if (state.thread_index == 0) {
      rmm::logger().log(spdlog::level::info, "------ Start of Benchmark -----");
      mr_ = factory_();
    }
  }

  /// Destroy the memory resource and count any unallocated memory
  void TearDown(const ::benchmark::State& state)
  {
    if (state.thread_index == 0) {
      rmm::logger().log(spdlog::level::info, "------ End of Benchmark -----");
      // clean up any leaked allocations
      std::size_t total_leaked{0};
      std::size_t num_leaked{0};
      for (auto const& ptr_alloc : allocation_map) {
        auto alloc = ptr_alloc.second;
        num_leaked++;
        total_leaked += alloc.size;
        mr_->deallocate(alloc.p, alloc.size, 0);
      }
      if (num_leaked > 0)
        std::cout << "LOG shows leak of " << num_leaked << " allocations of " << total_leaked
                  << " total bytes\n";
      allocation_map.clear();
      mr_.reset();
    }
  }

  /// Run the replay benchmark
  void operator()(::benchmark::State& state)
  {
    SetUp(state);

    auto const& my_events = events_.at(state.thread_index);

    for (auto _ : state) {
      std::for_each(my_events.begin(), my_events.end(), [&state, this](auto e) {
        // ensure correct ordering between threads
        std::unique_lock<std::mutex> lock{event_mutex};
        if (event_index != e.index) {
          cv.wait(lock, [&]() { return event_index == e.index; });
        }

        if (rmm::detail::action::ALLOCATE == e.act) {
          auto p = mr_->allocate(e.size);
          set_allocation(allocation_map, e.pointer, allocation{p, e.size});
        } else {
          auto a = remove_allocation(allocation_map, e.pointer);
          mr_->deallocate(a.p, e.size);
        }

        event_index++;
        cv.notify_all();
      });
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
                          [](auto const& e) {
                            return (e.stream == cudaStreamDefault) or
                                   (e.stream == reinterpret_cast<uintptr_t>(cudaStreamPerThread));
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

void declare_benchmark(std::string name,
                       std::vector<std::vector<rmm::detail::event>> const& per_thread_events,
                       std::size_t num_threads)
{
  if (name == "cuda")
    benchmark::RegisterBenchmark("CUDA Resource", replay_benchmark(&make_cuda, per_thread_events))
      ->Unit(benchmark::kMillisecond)
      ->Threads(num_threads);
  else if (name == "binning")
    benchmark::RegisterBenchmark("Binning Resource",
                                 replay_benchmark(&make_binning, per_thread_events))
      ->Unit(benchmark::kMillisecond)
      ->Threads(num_threads);
  else if (name == "pool")
    benchmark::RegisterBenchmark("Pool Resource", replay_benchmark(&make_pool, per_thread_events))
      ->Unit(benchmark::kMillisecond)
      ->Threads(num_threads);
  else if (name == "arena")
    benchmark::RegisterBenchmark("Arena Resource", replay_benchmark(&make_arena, per_thread_events))
      ->Unit(benchmark::kMillisecond)
      ->Threads(num_threads);
  else
    std::cout << "Error: invalid memory_resource name: " << name << "\n";
}

// Usage: REPLAY_BENCHMARK -f "path/to/log/file"
int main(int argc, char** argv)
{
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

  auto per_thread_events = parse_per_thread_events(filename);

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  std::cout << "Using CUDA per-thread default stream.\n";
#endif

  std::cout << "Total Events: "
            << std::accumulate(
                 per_thread_events.begin(),
                 per_thread_events.end(),
                 0,
                 [](std::size_t accum, auto const& events) { return accum + events.size(); })
            << std::endl;

  for (std::size_t t = 0; t < per_thread_events.size(); ++t) {
    std::cout << "Thread " << t << ": " << per_thread_events[t].size() << " events\n";
    if (args["verbose"].as<bool>()) {
      for (auto const& e : per_thread_events[t]) {
        std::cout << e << std::endl;
      }
    }
  }

  auto const num_threads = per_thread_events.size();

  // Uncomment to enable / change default log level
  // rmm::logger().set_level(spdlog::level::trace);

  if (args.count("resource") > 0) {
    std::string mr_name = args["resource"].as<std::string>();
    declare_benchmark(mr_name, per_thread_events, num_threads);
  } else {
    std::array<std::string, 4> mrs{"pool", "arena", "binning", "cuda"};
    std::for_each(
      std::cbegin(mrs), std::cend(mrs), [&per_thread_events, &num_threads](auto const& s) {
        declare_benchmark(s, per_thread_events, num_threads);
      });
  }

  ::benchmark::RunSpecifiedBenchmarks();
}
