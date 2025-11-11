/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @brief Doxygen group definitions
 */

// This header is only processed by doxygen and does
// not need to be included in any source file.
// Below are the main groups that doxygen uses to build
// the Modules page in the specified order.
//
// To add a new API to an existing group, just use the
// @ingroup tag to the API's doxygen comment.
// Add a new group by first specifying in the hierarchy below.

/**
 * @namespace rmm
 * @brief RAPIDS Memory Manager - The top level namespace for all RMM functionality
 *
 * The rmm namespace provides a comprehensive set of memory management
 * utilities for CUDA applications, including memory resources, CUDA stream
 * management, device-side data containers, and memory allocation utilities.
 */

/**
 * @namespace rmm::mr
 * @brief Memory Resource classes and adaptors
 *
 * The rmm::mr namespace contains all base memory resource classes that
 * implement various CUDA memory allocation strategies, adaptors for
 * suballocation such as pool and arena adaptors, and adaptors that add
 * functionality such as logging, alignment, and statistics tracking to
 * existing memory resources.
 */

/**
 * @defgroup memory_resources Memory Resources
 * @{
 *   @defgroup device_memory_resources Device Memory Resources
 *   @defgroup host_memory_resources Host Memory Resources
 *   @defgroup device_resource_adaptors Device Resource Adaptors
 * @}
 * @defgroup cuda_device_management CUDA Device Management
 * @defgroup cuda_streams CUDA Streams
 * @defgroup data_containers Data Containers
 * @defgroup errors Errors
 * @defgroup logging Logging
 * @defgroup thrust_integrations Thrust Integrations
 * @defgroup utilities Utilities
 */
