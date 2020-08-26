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

#include <benchmarks/utilities/cxxopts.hpp>
#include <benchmarks/utilities/log_parser.hpp>

#include <rmm/detail/error.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

#include <benchmark/benchmark.h>

#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include "rmm/mr/device/device_memory_resource.hpp"

/// MR factory functions
inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

inline auto make_pool()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_cuda());
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
  std::shared_ptr<rmm::mr::device_memory_resource> mr_{};
  std::vector<std::vector<rmm::detail::event>> const& events_{};

  /**
   * @brief Construct a `replay_benchmark` from a list of events and
   * set of arguments forwarded to the MR constructor.
   *
   * @param events The set of allocation events to replay
   * @param args Variable number of arguments forward to the constructor of MR
   */
  replay_benchmark(MRFactoryFunc factory,
                   std::vector<std::vector<rmm::detail::event>> const& events)
    : mr_{factory()}, events_{events}
  {
  }

  void operator()(benchmark::State& state)
  {
    // Maps a pointer from the event log to an active allocation
    std::unordered_map<uintptr_t, allocation> allocation_map(events_.size());

    auto const& my_events = events_.at(state.thread_index);

    for (auto _ : state) {
      std::for_each(my_events.begin(), my_events.end(), [&allocation_map, &state, this](auto e) {
        if (rmm::detail::action::ALLOCATE == e.act) {
          auto p                    = mr_->allocate(e.size);
          allocation_map[e.pointer] = allocation{p, e.size};
        } else {
          auto a = allocation_map[e.pointer];
          mr_->deallocate(a.p, e.size);
        }
      });
    }
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
                   return std::vector<event>(all_events.cbegin() + begin,
                                             all_events.cbegin() + end);
                 });

  return per_thread_events;
}

// Usage: REPLAY_BENCHMARK -f "path/to/log/file"
int main(int argc, char** argv)
{
  // benchmark::Initialize will remove GBench command line arguments it
  // recognizes and leave any remaining arguments
  ::benchmark::Initialize(&argc, argv);

  // Parse for replay arguments:
  cxxopts::Options options("RMM Replay Benchmark",
                           "Replays and benchmarks allocation activity captured from RMM logging.");

  options.add_options()("f,file", "Name of RMM log file.", cxxopts::value<std::string>());
  options.add_options()("v,verbose",
                        "Enable verbose printing of log events",
                        cxxopts::value<bool>()->default_value("false"));

  auto args = options.parse(argc, argv);

  if (args.count("file") == 0) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

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

  benchmark::RegisterBenchmark("CUDA Resource", replay_benchmark{&make_cuda, per_thread_events})
    ->Unit(benchmark::kMillisecond)
    ->Threads(num_threads);

  benchmark::RegisterBenchmark("Pool Resource", replay_benchmark(&make_pool, per_thread_events))
    ->Unit(benchmark::kMillisecond)
    ->Threads(num_threads);

  benchmark::RegisterBenchmark("Binning Resource",
                               replay_benchmark(&make_binning, per_thread_events))
    ->Unit(benchmark::kMillisecond)
    ->Threads(num_threads);

  ::benchmark::RunSpecifiedBenchmarks();
}
