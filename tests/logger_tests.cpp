/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <benchmarks/utilities/log_parser.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

class raii_restore_env {
 public:
  raii_restore_env(char const* name) : name_(name)
  {
    auto* const value_or_null = getenv(name);
    if (value_or_null != nullptr) {
      value_  = value_or_null;
      is_set_ = true;
    }
  }

  ~raii_restore_env()
  {
    if (is_set_) {
      setenv(name_.c_str(), value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  raii_restore_env(raii_restore_env const&) = default;
  raii_restore_env& operator=(raii_restore_env const&) = default;
  raii_restore_env(raii_restore_env&&)                 = default;
  raii_restore_env& operator=(raii_restore_env&&) = default;

 private:
  std::string name_{};
  std::string value_{};
  bool is_set_{false};
};

/**
 * @brief Verifies the specified log file contains the expected events.
 *
 * Events in the log file are expected to occur in the same order as in `expected_events`.
 *
 * @note: This function accounts for the fact that `device_memory_resource` automatically pads
 * allocations to a multiple of 8 bytes by rounding up the expected allocation sizes to a multiple
 * of 8.
 *
 * @param filename Name of CSV log file generated from `logging_resource_adaptor`
 * @param expected_events List of expected (de)allocation events
 */
void expect_log_events(std::string const& filename,
                       std::vector<rmm::detail::event> const& expected_events)
{
  auto actual_events = rmm::detail::parse_csv(filename);

  std::equal(expected_events.begin(),
             expected_events.end(),
             actual_events.begin(),
             [](auto expected, auto actual) {
               // We don't test the logged thread id since it may be different from what we record.
               // The actual value doesn't matter so long as events from different threads have
               // different ids
               // EXPECT_EQ(expected.thread_id, actual.thread_id);
               // EXPECT_EQ(expected.stream, actual.stream);
               EXPECT_EQ(expected.act, actual.act);
               // device_memory_resource automatically pads an allocation to a multiple of 8 bytes
               EXPECT_EQ(rmm::detail::align_up(expected.size, 8), actual.size);
               EXPECT_EQ(expected.pointer, actual.pointer);
               return true;
             });
}

TEST(Adaptor, FilenameConstructor)
{
  std::string filename{"logs/test1.txt"};
  rmm::mr::cuda_memory_resource upstream;
  rmm::mr::logging_resource_adaptor<rmm::mr::cuda_memory_resource> log_mr{&upstream, filename};

  auto const size0{100};
  auto const size1{42};

  auto* ptr0 = log_mr.allocate(size0);
  auto* ptr1 = log_mr.allocate(size1);
  log_mr.deallocate(ptr0, size0);
  log_mr.deallocate(ptr1, size1);
  log_mr.flush();

  using rmm::detail::action;
  using rmm::detail::event;

  std::vector<event> expected_events{{action::ALLOCATE, size0, ptr0},
                                     {action::ALLOCATE, size1, ptr1},
                                     {action::FREE, size0, ptr0},
                                     {action::FREE, size1, ptr1}};

  expect_log_events(filename, expected_events);
}

TEST(Adaptor, MultiSinkConstructor)
{
  std::string filename1{"logs/test_multi_1.txt"};
  std::string filename2{"logs/test_multi_2.txt"};
  rmm::mr::cuda_memory_resource upstream;

  auto file_sink1 = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename1, true);
  auto file_sink2 = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename2, true);

  rmm::mr::logging_resource_adaptor<rmm::mr::cuda_memory_resource> log_mr{&upstream,
                                                                          {file_sink1, file_sink2}};

  auto const size0{100};
  auto const size1{42};

  auto* ptr0 = log_mr.allocate(size0);
  auto* ptr1 = log_mr.allocate(size1);
  log_mr.deallocate(ptr0, size0);
  log_mr.deallocate(ptr1, size1);
  log_mr.flush();

  using rmm::detail::action;
  using rmm::detail::event;

  std::vector<event> expected_events{{action::ALLOCATE, size0, ptr0},
                                     {action::ALLOCATE, size1, ptr1},
                                     {action::FREE, size0, ptr0},
                                     {action::FREE, size1, ptr1}};

  expect_log_events(filename1, expected_events);
  expect_log_events(filename2, expected_events);
}

TEST(Adaptor, Factory)
{
  std::string filename{"logs/test2.txt"};
  rmm::mr::cuda_memory_resource upstream;

  auto log_mr = rmm::mr::make_logging_adaptor(&upstream, filename);

  auto const size0{99};
  auto const size1{42};

  auto* ptr0 = log_mr.allocate(size0);
  log_mr.deallocate(ptr0, size0);
  auto* ptr1 = log_mr.allocate(size1);
  log_mr.deallocate(ptr1, size1);
  log_mr.flush();

  using rmm::detail::action;
  using rmm::detail::event;

  std::vector<event> expected_events{{action::ALLOCATE, size0, ptr0},
                                     {action::FREE, size0, ptr0},
                                     {action::ALLOCATE, size1, ptr1},
                                     {action::FREE, size1, ptr1}};

  expect_log_events(filename, expected_events);
}

TEST(Adaptor, EnvironmentPath)
{
  rmm::mr::cuda_memory_resource upstream;

  // restore the original value (or unset) after test
  raii_restore_env old_env("RMM_LOG_FILE");

  unsetenv("RMM_LOG_FILE");

  // expect logging adaptor to fail if RMM_LOG_FILE is unset
  EXPECT_THROW(rmm::mr::make_logging_adaptor(&upstream), rmm::logic_error);

  std::string filename("logs/envtest.txt");

  setenv("RMM_LOG_FILE", filename.c_str(), 1);

  // use log file location specified in environment variable RMM_LOG_FILE
  auto log_mr = rmm::mr::make_logging_adaptor(&upstream);

  auto const size{100};

  auto* ptr = log_mr.allocate(size);
  log_mr.deallocate(ptr, size);

  log_mr.flush();

  using rmm::detail::action;
  using rmm::detail::event;

  std::vector<event> expected_events{
    {action::ALLOCATE, size, ptr},
    {action::FREE, size, ptr},
  };

  expect_log_events(filename, expected_events);
}

TEST(Adaptor, STDOUT)
{
  testing::internal::CaptureStdout();

  rmm::mr::cuda_memory_resource upstream;

  auto log_mr = rmm::mr::make_logging_adaptor(&upstream, std::cout);

  auto const size{100};

  auto* ptr = log_mr.allocate(size);
  log_mr.deallocate(ptr, size);

  std::string output = testing::internal::GetCapturedStdout();
  std::string header = output.substr(0, output.find('\n'));
  ASSERT_EQ(header, log_mr.header());
}

TEST(Adaptor, STDERR)
{
  testing::internal::CaptureStderr();

  rmm::mr::cuda_memory_resource upstream;

  auto log_mr = rmm::mr::make_logging_adaptor(&upstream, std::cerr);

  auto const size{100};

  auto* ptr = log_mr.allocate(size);
  log_mr.deallocate(ptr, size);

  std::string output = testing::internal::GetCapturedStderr();
  std::string header = output.substr(0, output.find('\n'));
  ASSERT_EQ(header, log_mr.header());
}
