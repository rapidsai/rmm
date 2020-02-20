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

#include <benchmark/benchmark.h>

static void BM_StringCreation(benchmark::State& state) {
  for (auto _ : state) std::string empty_string;
}
// Register the function as a benchmark
BENCHMARK(BM_StringCreation);

// Define another benchmark
static void BM_StringCopy(benchmark::State& state) {
  std::string x = "hello";
  for (auto _ : state) std::string copy(x);
}
BENCHMARK(BM_StringCopy);

void parse_csv(std::string const& filename) {
  rapidcsv::Document csv(filename);

  std::vector<std::string> actions = csv.GetColumn<std::string>("Action");
  std::vector<std::size_t> sizes = csv.GetColumn<std::size_t>("Size");
}

int main(int argc, char** argv) {
  // benchmark::Initialize will remove command line arguments it recognizes and
  // leave any remaining arguments
  ::benchmark::Initialize(&argc, argv);

  // Parse for replay arguments:
  cxxopts::Options options(
      "RMM Replay Benchmark",
      "Replays and benchmarks allocation activity captured from RMM logging.");

  options.add_options()("f,file", "Name of RMM log file.",
                        cxxopts::value<std::string>());

  auto result = options.parse(argc, argv);

  if (result.count("file")) {
    auto filename = result["file"].as<std::string>();
    parse_csv(filename);
  } else {
    // throw std::runtime_error{"No log filename specified."};
  }

  ::benchmark::RunSpecifiedBenchmarks();
}