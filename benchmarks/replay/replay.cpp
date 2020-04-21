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

#include "cxxopts.hpp"
#include "rapidcsv.h"

#include <rmm/mr/device/cnmem_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <benchmark/benchmark.h>
#include <thrust/iterator/zip_iterator.h>
#include <memory>
#include <stdexcept>
#include <string>

enum class action : bool { ALLOCATE, FREE };

/**
 * @brief Represents an allocation event
 *
 */
struct event {
  action act;         ///< Indicates if the event is an allocation or a free
  std::size_t size;   ///< The size of the memory allocated or free'd
  uintptr_t pointer;  ///< The pointer returned from an allocation, or the
                      ///< pointer free'd
};

/**
 * @brief Parses the RMM log file specifed by `filename` for consumption by the
 * replay benchmark.
 *
 * @param filename Name of the RMM log file
 * @return Vector of events for consumption by replay benchmark
 */
std::vector<event>
parse_csv(std::string const& filename) {
  rapidcsv::Document csv(filename);

  std::vector<std::string> actions  = csv.GetColumn<std::string>("Action");
  std::vector<std::size_t> sizes    = csv.GetColumn<std::size_t>("Size");
  std::vector<std::string> pointers = csv.GetColumn<std::string>("Pointer");

  if ((sizes.size() != actions.size()) or (sizes.size() != pointers.size())) {
    throw std::runtime_error{"Size mismatch in actions, sizes, or pointers."};
  }

  std::vector<event> events(sizes.size());

  auto zipped_begin =
    thrust::make_zip_iterator(thrust::make_tuple(actions.begin(), sizes.begin(), pointers.begin()));
  auto zipped_end = zipped_begin + sizes.size();

  std::transform(zipped_begin,
                 zipped_end,
                 events.begin(),
                 [](thrust::tuple<std::string, std::size_t, std::string> const& t) {
                   // Convert "allocate" or "free" string into `action` enum
                   action a = (thrust::get<0>(t) == "allocate") ? action::ALLOCATE : action::FREE;
                   std::size_t size = thrust::get<1>(t);

                   // Convert pointer string into an integer
                   uintptr_t p = std::stoll(thrust::get<2>(t), nullptr, 16);
                   return event{a, size, p};
                 });

  return events;
}

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
template <typename MR>
struct replay_benchmark {
  std::unique_ptr<MR> mr_{};
  std::vector<event> const& events_{};

  /**
   * @brief Construct a `replay_benchmark` from a list of events and
   * set of arguments forwarded to the MR constructor.
   *
   * @param events The set of allocation events to replay
   * @param args Variable number of arguments forward to the constructor of MR
   */
  template <typename... Args>
  replay_benchmark(std::vector<event> const& events, Args&&... args)
    : mr_{new MR{std::forward<Args>(args)...}}, events_{events} {}

  void
  operator()(benchmark::State& state) {
    // Maps a pointer from the event log to an active allocation
    std::unordered_map<uintptr_t, allocation> allocation_map(events_.size());

    for (auto _ : state) {
      std::for_each(events_.begin(), events_.end(), [&allocation_map, &state, this](event e) {
        if (action::ALLOCATE == e.act) {
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

// Usage: REPLAY_BENCHMARK -f "path/to/log/file"
int
main(int argc, char** argv) {
  // benchmark::Initialize will remove GBench command line arguments it
  // recognizes and leave any remaining arguments
  ::benchmark::Initialize(&argc, argv);

  // Parse for replay arguments:
  cxxopts::Options options("RMM Replay Benchmark",
                           "Replays and benchmarks allocation activity captured from RMM logging.");

  options.add_options()("f,file", "Name of RMM log file.", cxxopts::value<std::string>());

  auto result = options.parse(argc, argv);

  // Parse the log file
  if (result.count("file")) {
    auto filename = result["file"].as<std::string>();
    auto events   = parse_csv(filename);

    benchmark::RegisterBenchmark("CUDA Resource",
                                 replay_benchmark<rmm::mr::cuda_memory_resource>{events})
      ->Unit(benchmark::kMillisecond);

    benchmark::RegisterBenchmark("CNMEM Resource",
                                 replay_benchmark<rmm::mr::cnmem_memory_resource>(events, 0u))
      ->Unit(benchmark::kMillisecond);

    ::benchmark::RunSpecifiedBenchmarks();
  } else {
    throw std::runtime_error{"No log filename specified."};
  }
}
