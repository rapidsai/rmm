/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * IPC Pool Importer Example
 *
 * This example demonstrates how to import a shared CUDA memory pool from another
 * process and use it for multi-GPU peer-to-peer writes. The importer:
 *   1. Connects to the exporter via Unix domain socket
 *   2. Receives the pool handle and pointer metadata
 *   3. Imports the pool using cudaMemPoolImportFromShareableHandle()
 *   4. Imports the pointer using cudaMemPoolImportPointer()
 *   5. Spawns multiple threads, each writing to a disjoint region from different GPUs
 */

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/runtime_capabilities.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>

#include <cuda_runtime.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr char const* default_socket_path = "/tmp/rmm_ipc_pool.sock";

/**
 * @brief Connect to a Unix domain socket server
 */
int connect_socket(char const* path)
{
  int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  RMM_EXPECTS(fd >= 0, "Failed to create socket");

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path);

  RMM_EXPECTS(::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) >= 0,
              "Failed to connect to socket");

  return fd;
}

/**
 * @brief Receive all bytes from a socket
 */
void recv_all(int sock, void* data, std::size_t n)
{
  char* ptr = static_cast<char*>(data);
  while (n > 0) {
    ssize_t received = ::recv(sock, ptr, n, MSG_WAITALL);
    RMM_EXPECTS(received > 0, "Failed to receive data");
    ptr += received;
    n -= static_cast<std::size_t>(received);
  }
}

/**
 * @brief Send all bytes over a socket
 */
void send_all(int sock, void const* data, std::size_t n)
{
  char const* ptr = static_cast<char const*>(data);
  while (n > 0) {
    ssize_t sent = ::send(sock, ptr, n, 0);
    RMM_EXPECTS(sent >= 0, "Failed to send data");
    ptr += sent;
    n -= static_cast<std::size_t>(sent);
  }
}

/**
 * @brief Receive a file descriptor over a Unix domain socket using SCM_RIGHTS
 */
int recv_fd(int sock)
{
  char buf  = 0;
  iovec iov = {&buf, 1};
  char cmsgbuf[CMSG_SPACE(sizeof(int))];
  std::memset(cmsgbuf, 0, sizeof(cmsgbuf));

  msghdr msg         = {};
  msg.msg_iov        = &iov;
  msg.msg_iovlen     = 1;
  msg.msg_control    = cmsgbuf;
  msg.msg_controllen = sizeof(cmsgbuf);

  RMM_EXPECTS(::recvmsg(sock, &msg, 0) >= 0, "Failed to receive message");

  cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  RMM_EXPECTS(cmsg != nullptr && cmsg->cmsg_type == SCM_RIGHTS, "Did not receive file descriptor");

  int fd = -1;
  std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
  return fd;
}

/**
 * @brief CUDA kernel to fill a buffer with a pattern
 */
__global__ void fill_pattern(unsigned int* ptr, unsigned int pattern, std::size_t count)
{
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) { ptr[idx] = pattern; }
}

/**
 * @brief Worker thread function that writes to a region of the imported buffer
 *
 * Each thread:
 * 1. Sets its assigned GPU as current device
 * 2. Enables peer access to the target GPU
 * 3. Allocates a local buffer and fills it with a pattern
 * 4. Copies to a disjoint region of the imported buffer via P2P
 */
void writer_thread(int writer_gpu,
                   int target_gpu,
                   void* imported_ptr,
                   std::size_t offset,
                   std::size_t bytes,
                   unsigned int pattern)
{
  // Set this thread's GPU
  RMM_CUDA_TRY(cudaSetDevice(writer_gpu));
  RMM_CUDA_TRY(cudaFree(nullptr));  // Context warmup

  // Check and enable peer access
  int can_access = 0;
  RMM_CUDA_TRY(cudaDeviceCanAccessPeer(&can_access, writer_gpu, target_gpu));
  if (can_access == 0) {
    std::fprintf(stderr,
                 "[writer %d] Warning: No P2P access to target GPU %d, using staged copy\n",
                 writer_gpu,
                 target_gpu);
  } else {
    cudaError_t err = cudaDeviceEnablePeerAccess(target_gpu, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) { RMM_CUDA_TRY(err); }
  }

  // Create stream for this thread
  rmm::cuda_stream stream{};

  // Allocate local source buffer using device_uvector
  std::size_t const num_elements = bytes / sizeof(unsigned int);
  rmm::device_uvector<unsigned int> src_buffer(num_elements, stream);

  // Fill with recognizable pattern using CUDA kernel
  int const threads_per_block = 256;
  int const num_blocks =
    static_cast<int>((num_elements + threads_per_block - 1) / threads_per_block);
  fill_pattern<<<num_blocks, threads_per_block, 0, stream.value()>>>(
    src_buffer.data(), pattern, num_elements);
  RMM_CUDA_TRY(cudaGetLastError());

  // Copy to the target region via P2P
  void* dst_region = static_cast<char*>(imported_ptr) + offset;
  RMM_CUDA_TRY(cudaMemcpyPeerAsync(
    dst_region, target_gpu, src_buffer.data(), writer_gpu, bytes, stream.value()));

  stream.synchronize();

  std::fprintf(stderr,
               "[writer %d] Wrote pattern 0x%08X to offset %zu (%zu bytes)\n",
               writer_gpu,
               pattern,
               offset,
               bytes);
}

void print_usage(char const* prog_name)
{
  std::fprintf(stderr,
               "Usage: %s <target_gpu> <writer_gpu_0> [writer_gpu_1] [writer_gpu_2] [writer_gpu_3] "
               "[socket_path]\n"
               "  target_gpu:     GPU device ID where the shared pool resides\n"
               "  writer_gpu_N:   GPU device IDs that will write to the buffer (1-4 GPUs)\n"
               "  socket_path:    Unix socket path (default: %s)\n",
               prog_name,
               default_socket_path);
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 3) {
    print_usage(argv[0]);
    return 1;
  }

  int const target_gpu = std::atoi(argv[1]);

  // Collect writer GPUs (variable number, 1-4)
  std::vector<int> writers;
  int arg_idx = 2;
  while (arg_idx < argc && arg_idx < 6) {
    // Check if this looks like a GPU ID (numeric) or socket path
    char* end_ptr  = nullptr;
    long const val = std::strtol(argv[arg_idx], &end_ptr, 10);
    if (*end_ptr != '\0') {
      break;  // Not a number, must be socket path
    }
    writers.push_back(static_cast<int>(val));
    ++arg_idx;
  }

  if (writers.empty()) {
    std::fprintf(stderr, "Error: At least one writer GPU must be specified\n");
    print_usage(argv[0]);
    return 1;
  }

  char const* socket_path = (arg_idx < argc) ? argv[arg_idx] : default_socket_path;

  std::fprintf(stderr,
               "[importer] Target GPU: %d, Writer GPUs: %zu, Socket: %s\n",
               target_gpu,
               writers.size(),
               socket_path);

  // Connect to exporter
  int sock = connect_socket(socket_path);
  std::fprintf(stderr, "[importer] Connected to exporter\n");

  // Receive metadata from exporter
  int recv_target_gpu = -1;
  std::size_t bytes   = 0;
  recv_all(sock, &recv_target_gpu, sizeof(recv_target_gpu));
  recv_all(sock, &bytes, sizeof(bytes));
  int pool_fd = recv_fd(sock);

  cudaMemPoolPtrExportData export_data{};
  recv_all(sock, &export_data, sizeof(export_data));

  std::fprintf(stderr,
               "[importer] Received: target_gpu=%d, bytes=%zu, pool_fd=%d\n",
               recv_target_gpu,
               bytes,
               pool_fd);

  // Verify target GPU matches
  if (recv_target_gpu != target_gpu) {
    std::fprintf(stderr,
                 "Error: Target GPU mismatch (exporter=%d, importer=%d)\n",
                 recv_target_gpu,
                 target_gpu);
    return 1;
  }

  // Import the pool (using the first writer GPU's context)
  RMM_CUDA_TRY(cudaSetDevice(writers[0]));
  RMM_CUDA_TRY(cudaFree(nullptr));  // Context warmup

  cudaMemPool_t imported_pool{};
  // Note: For POSIX FD handle type, pass the FD value cast to void*, not a pointer to it
  RMM_CUDA_TRY(cudaMemPoolImportFromShareableHandle(
    &imported_pool,
    reinterpret_cast<void*>(static_cast<std::intptr_t>(pool_fd)),
    cudaMemHandleTypePosixFileDescriptor,
    0));
  ::close(pool_fd);

  std::fprintf(stderr, "[importer] Imported pool handle\n");

  // Import the pointer allocation
  void* imported_ptr = nullptr;
  RMM_CUDA_TRY(cudaMemPoolImportPointer(&imported_ptr, imported_pool, &export_data));

  std::fprintf(stderr, "[importer] Imported pointer: %p\n", imported_ptr);

  // Create a view memory resource wrapping the imported pool (for demonstration)
  // This could be used for further allocations from the imported pool
  rmm::mr::cuda_async_view_memory_resource view_mr{imported_pool};
  std::fprintf(stderr, "[importer] Created view memory resource for imported pool\n");

  // Calculate chunk sizes for each writer
  std::size_t const num_writers = writers.size();
  std::size_t const chunk_size  = bytes / num_writers;
  std::size_t const last_chunk  = bytes - (num_writers - 1) * chunk_size;

  // Patterns for each writer (easily identifiable in memory dumps)
  std::vector<unsigned int> patterns = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};

  // Launch writer threads
  std::vector<std::thread> threads;
  threads.reserve(num_writers);

  for (std::size_t i = 0; i < num_writers; ++i) {
    std::size_t const offset       = i * chunk_size;
    std::size_t const thread_bytes = (i == num_writers - 1) ? last_chunk : chunk_size;
    unsigned int const pattern     = patterns[i % patterns.size()];

    threads.emplace_back(
      writer_thread, writers[i], target_gpu, imported_ptr, offset, thread_bytes, pattern);
  }

  // Wait for all writers to complete
  for (auto& t : threads) {
    t.join();
  }

  std::fprintf(stderr, "[importer] All %zu writer threads completed\n", num_writers);

  // Signal completion to exporter
  char done_signal = 'D';
  send_all(sock, &done_signal, sizeof(done_signal));
  ::close(sock);

  // Cleanup: destroy imported pool handle
  // Note: The backing allocation lifetime is managed by the exporter.
  // Use non-throwing version since the pool may already be invalidated
  // if the exporter cleaned up before us.
  cudaError_t cleanup_err = cudaMemPoolDestroy(imported_pool);
  if (cleanup_err != cudaSuccess && cleanup_err != cudaErrorInvalidValue) {
    std::fprintf(stderr,
                 "[importer] Warning: cudaMemPoolDestroy failed: %s\n",
                 cudaGetErrorString(cleanup_err));
  }

  std::fprintf(stderr, "[importer] Cleanup complete\n");

  return 0;
}
