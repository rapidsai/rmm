/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

extern "C" void const* current_state_a_map();
extern "C" void const* current_state_b_map();
extern "C" void const* previous_state_map();

int main()
{
  if (current_state_a_map() != current_state_b_map()) { return 1; }
  if (current_state_a_map() == previous_state_map()) { return 2; }
  return 0;
}
