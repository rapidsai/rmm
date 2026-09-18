/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/error.hpp>

#include <string>

RMM_NAMESPACE_BEGIN

bad_alloc::bad_alloc(const char* msg) : _what{std::string{std::bad_alloc::what()} + ": " + msg} {}

bad_alloc::bad_alloc(std::string const& msg) : bad_alloc{msg.c_str()} {}

const char* bad_alloc::what() const noexcept { return _what.c_str(); }

out_of_memory::out_of_memory(const char* msg) : bad_alloc{std::string{"out_of_memory: "} + msg} {}

out_of_memory::out_of_memory(std::string const& msg) : out_of_memory{msg.c_str()} {}

RMM_NAMESPACE_END
