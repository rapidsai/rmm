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
#include <string>

enum class action : bool { ALLOCATE, FREE };

/**
 * @brief Stores the contents of a parsed log
 *
 * Holds 3 vectors of length `n`, where `n` is the number of actions in the log
 * - actions: Indicates if action `i` is an allocation or a deallocation
 * - sizes: Indicates the size of the action `i`
 * - pointers: For an allocation, the pointer returned, for a free, the pointer
 *   freed
 */
struct parsed_log {
  parsed_log(std::vector<action>&& a, std::vector<std::size_t>&& s,
             std::vector<uintptr_t>&& p)
      : actions{std::move(a)}, sizes{std::move(s)}, pointers{std::move(p)} {
    if ((actions.size() != sizes.size()) or
        (actions.size() != pointers.size())) {
      throw std::runtime_error{
          "Size mismatch between actions, sizes, pointers."};
    }
  }
  std::vector<action> actions{};
  std::vector<std::size_t> sizes{};
  std::vector<uintptr_t> pointers{};
};

/**
 * @brief Parses the RMM log file specifed by `filename` for consumption by the
 * replay benchmark.
 *
 * @param filename Name of the RMM log file
 * @return parsed_log The logfile parsed into a set of actions that can be
 * consumed by the replay benchmark.
 */
parsed_log parse_csv(std::string const& filename) {
  rapidcsv::Document csv(filename);

  std::vector<std::size_t> sizes = csv.GetColumn<std::size_t>("Size");

  // Convert action strings to enum to reduce overhead of processing actions in
  // benchmark
  std::vector<std::string> actions_as_string =
      csv.GetColumn<std::string>("Action");
  std::vector<action> actions(actions_as_string.size());
  std::transform(actions_as_string.begin(), actions_as_string.end(),
                 actions.begin(), [](std::string const& s) {
                   return (s == "allocate") ? action::ALLOCATE : action::FREE;
                 });

  // Convert address string to uintptr_t
  // E.g., 0x7fb3c446f000 -> 140410068856832
  std::vector<std::string> pointers_as_string =
      csv.GetColumn<std::string>("Pointer");
  std::vector<uintptr_t> pointers(pointers_as_string.size());
  std::transform(
      pointers_as_string.begin(), pointers_as_string.end(), pointers.begin(),
      [](std::string const& s) { return std::stoll(s, nullptr, 16); });

  return parsed_log{std::move(actions), std::move(sizes), std::move(pointers)};
}

struct allocation {
  allocation() = default;
  allocation(void* p_, std::size_t size_) : p{p_}, size{size_} {}
  void* p{};
  std::size_t size{};
};

template <typename MR>
struct replay_benchmark {
  MR mr_{};
  parsed_log const& log_{};

  replay_benchmark(MR const& mr, parsed_log const& log) : mr_{mr}, log_{log} {}

  void operator()(benchmark::State& state) {
    std::unordered_map<uintptr_t, allocation> allocation_map;

    auto const& actions = log_.actions;
    auto const& sizes = log_.sizes;
    auto const& pointers = log_.pointers;

    for (auto _ : state) {
      for (int i = 0; i < actions.size(); ++i) {
        if (action::ALLOCATE == actions[i]) {
          auto p = mr_.allocate(sizes[i]);
          allocation_map[pointers[i]] = allocation{p, sizes[i]};
        } else {
          auto a = allocation_map[pointers[i]];
          mr_.deallocate(a.p, a.size);
        }
      }
    }
  }
};

int main(int argc, char** argv) {
  // benchmark::Initialize will remove GBench command line arguments it
  // recognizes and leave any remaining arguments
  ::benchmark::Initialize(&argc, argv);

  // Parse for replay arguments:
  cxxopts::Options options(
      "RMM Replay Benchmark",
      "Replays and benchmarks allocation activity captured from RMM logging.");

  options.add_options()("f,file", "Name of RMM log file.",
                        cxxopts::value<std::string>());

  auto result = options.parse(argc, argv);

  // Parse the log file
  if (result.count("file")) {
    auto filename = result["file"].as<std::string>();
    auto parsed_log = parse_csv(filename);

    std::cout << "Number of actions: " << parsed_log.actions.size()
              << std::endl;

    benchmark::RegisterBenchmark(
        "CUDA Resource",
        replay_benchmark<rmm::mr::cuda_memory_resource>{
            rmm::mr::cuda_memory_resource{}, parsed_log})
        ->Unit(benchmark::kMillisecond);

    /*
        benchmark::RegisterBenchmark(
            "CNMEM Resource",
            replay_benchmark<rmm::mr::cnmem_memory_resource>{
                rmm::mr::cnmem_memory_resource{}, parsed_log})
            ->Unit(benchmark::kMillisecond);
            */
    ::benchmark::RunSpecifiedBenchmarks();
  } else {
    throw std::runtime_error{"No log filename specified."};
  }
}