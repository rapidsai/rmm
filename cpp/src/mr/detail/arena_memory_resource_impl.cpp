/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/detail/format.hpp>
#include <rmm/detail/logging_assert.hpp>
#include <rmm/logger.hpp>
#include <rmm/mr/detail/arena_memory_resource_impl.hpp>

#include <cuda_runtime_api.h>

namespace RMM_NAMESPACE {
namespace mr {
namespace detail {

arena_memory_resource_impl::arena_memory_resource_impl(device_async_resource_ref upstream_mr,
                                                       std::optional<std::size_t> arena_size,
                                                       bool dump_log_on_failure)
  : global_arena_{upstream_mr, arena_size}, dump_log_on_failure_{dump_log_on_failure}
{
  if (dump_log_on_failure_) {
    logger_ =
      std::make_shared<rapids_logger::logger>("arena_memory_dump", "rmm_arena_memory_dump.log");
    logger_->set_level(rapids_logger::level_enum::info);
  }
}

void* arena_memory_resource_impl::allocate(cuda::stream_ref stream,
                                           std::size_t bytes,
                                           std::size_t /*alignment*/)
{
  if (bytes <= 0) { return nullptr; }
  cuda_stream_view sv{stream.get()};
#ifdef RMM_ARENA_USE_SIZE_CLASSES
  bytes = rmm::mr::detail::arena::align_to_size_class(bytes);
#else
  bytes = rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
#endif
  auto& arena = get_arena(sv);

  {
    std::shared_lock lock(mtx_);
    void* pointer = arena.allocate_sync(bytes);
    if (pointer != nullptr) { return pointer; }
  }

  {
    std::unique_lock lock(mtx_);
    defragment();
    void* pointer = arena.allocate_sync(bytes);
    if (pointer == nullptr) {
      if (dump_log_on_failure_) { dump_memory_log(bytes); }
      auto const msg = std::string("Maximum pool size exceeded (failed to allocate ") +
                       rmm::detail::format_bytes(bytes) + "): No room in arena.";
      RMM_FAIL(msg.c_str(), rmm::out_of_memory);
    }
    return pointer;
  }
}

void arena_memory_resource_impl::deallocate(cuda::stream_ref stream,
                                            void* ptr,
                                            std::size_t bytes,
                                            std::size_t /*alignment*/) noexcept
{
  if (ptr == nullptr || bytes <= 0) { return; }
  cuda_stream_view sv{stream.get()};
#ifdef RMM_ARENA_USE_SIZE_CLASSES
  bytes = rmm::mr::detail::arena::align_to_size_class(bytes);
#else
  bytes = rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
#endif
  auto& arena = get_arena(sv);

  {
    std::shared_lock lock(mtx_);
    if (arena.deallocate(sv, ptr, bytes)) { return; }
  }

  {
    sv.synchronize_no_throw();
    std::unique_lock lock(mtx_);
    deallocate_from_other_arena(sv, ptr, bytes);
  }
}

void* arena_memory_resource_impl::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return allocate(cuda_stream_view{}, bytes, alignment);
}

void arena_memory_resource_impl::deallocate_sync(void* ptr,
                                                 std::size_t bytes,
                                                 std::size_t alignment) noexcept
{
  deallocate(cuda_stream_view{}, ptr, bytes, alignment);
}

void arena_memory_resource_impl::defragment()
{
  RMM_CUDA_TRY(cudaDeviceSynchronize());
  for (auto& thread_arena : thread_arenas_) {
    thread_arena.second->clean();
  }
  for (auto& stream_arena : stream_arenas_) {
    stream_arena.second.clean();
  }
}

void arena_memory_resource_impl::deallocate_from_other_arena(cuda_stream_view stream,
                                                             void* ptr,
                                                             std::size_t bytes)
{
  if (use_per_thread_arena(stream)) {
    for (auto const& thread_arena : thread_arenas_) {
      if (thread_arena.second->deallocate_sync(ptr, bytes)) { return; }
    }
  } else {
    for (auto& stream_arena : stream_arenas_) {
      if (stream_arena.second.deallocate_sync(ptr, bytes)) { return; }
    }
  }

  if (!global_arena_.deallocate_sync(ptr, bytes)) {
    if (use_per_thread_arena(stream)) {
      for (auto& stream_arena : stream_arenas_) {
        if (stream_arena.second.deallocate_sync(ptr, bytes)) { return; }
      }
    } else {
      for (auto const& thread_arena : thread_arenas_) {
        if (thread_arena.second->deallocate_sync(ptr, bytes)) { return; }
      }
    }
    RMM_FAIL("allocation not found");
  }
}

arena_memory_resource_impl::arena& arena_memory_resource_impl::get_arena(cuda_stream_view stream)
{
  if (use_per_thread_arena(stream)) { return get_thread_arena(); }
  return get_stream_arena(stream);
}

arena_memory_resource_impl::arena& arena_memory_resource_impl::get_thread_arena()
{
  auto const thread_id = std::this_thread::get_id();
  {
    std::shared_lock lock(map_mtx_);
    auto const iter = thread_arenas_.find(thread_id);
    if (iter != thread_arenas_.end()) { return *iter->second; }
  }
  {
    std::unique_lock lock(map_mtx_);
    auto thread_arena = std::make_shared<arena>(global_arena_);
    thread_arenas_.emplace(thread_id, thread_arena);
    thread_local detail::arena::arena_cleaner cleaner{thread_arena};
    return *thread_arena;
  }
}

arena_memory_resource_impl::arena& arena_memory_resource_impl::get_stream_arena(
  cuda_stream_view stream)
{
  RMM_LOGGING_ASSERT(!use_per_thread_arena(stream));
  {
    std::shared_lock lock(map_mtx_);
    auto const iter = stream_arenas_.find(stream.value());
    if (iter != stream_arenas_.end()) { return iter->second; }
  }
  {
    std::unique_lock lock(map_mtx_);
    stream_arenas_.emplace(stream.value(), global_arena_);
    return stream_arenas_.at(stream.value());
  }
}

void arena_memory_resource_impl::dump_memory_log(std::size_t bytes)
{
  logger_->info("**************************************************");
  logger_->info("Ran out of memory trying to allocate %s.", rmm::detail::format_bytes(bytes));
  logger_->info("**************************************************");
  logger_->info("Global arena:");
  global_arena_.dump_memory_log(logger_);
  logger_->flush();
}

bool arena_memory_resource_impl::use_per_thread_arena(cuda_stream_view stream)
{
  return stream.is_per_thread_default();
}

}  // namespace detail
}  // namespace mr
}  // namespace RMM_NAMESPACE
