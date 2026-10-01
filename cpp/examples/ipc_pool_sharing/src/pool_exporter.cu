/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * IPC Pool Exporter Example
 *
 * This example demonstrates how to create an IPC-enabled CUDA memory pool using
 * RMM's cuda_async_memory_resource and export it to another process. The exporter:
 *   1. Creates a shareable memory pool with POSIX file descriptor handle type
 *   2. Allocates a buffer from the pool
 *   3. Exports the pool handle and pointer metadata via Unix domain socket
 *   4. Keeps the allocation alive until the importer finishes
 */

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda_runtime.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr char const* default_socket_path = "/tmp/rmm_ipc_pool.sock";

/**
 * @brief Create a Unix domain socket server and listen for connections
 */
int make_server_socket(char const* path)
{
  int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  RMM_EXPECTS(fd >= 0, "Failed to create socket");

  // Remove any existing socket file
  ::unlink(path);

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path);

  RMM_EXPECTS(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) >= 0,
              "Failed to bind socket");
  RMM_EXPECTS(::listen(fd, 1) >= 0, "Failed to listen on socket");

  return fd;
}

/**
 * @brief Accept a client connection on the server socket
 */
int accept_client(int server_fd)
{
  int client_fd = ::accept(server_fd, nullptr, nullptr);
  RMM_EXPECTS(client_fd >= 0, "Failed to accept client connection");
  return client_fd;
}

/**
 * @brief Send a file descriptor over a Unix domain socket using SCM_RIGHTS
 */
void send_fd(int sock, int fd_to_send)
{
  char buf  = 'F';
  iovec iov = {&buf, 1};
  char cmsgbuf[CMSG_SPACE(sizeof(int))];
  std::memset(cmsgbuf, 0, sizeof(cmsgbuf));

  msghdr msg         = {};
  msg.msg_iov        = &iov;
  msg.msg_iovlen     = 1;
  msg.msg_control    = cmsgbuf;
  msg.msg_controllen = sizeof(cmsgbuf);

  cmsghdr* cmsg    = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type  = SCM_RIGHTS;
  cmsg->cmsg_len   = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(cmsg), &fd_to_send, sizeof(int));

  msg.msg_controllen = cmsg->cmsg_len;

  RMM_EXPECTS(::sendmsg(sock, &msg, 0) >= 0, "Failed to send file descriptor");
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

void print_usage(char const* prog_name)
{
  std::fprintf(stderr,
               "Usage: %s <target_gpu> <bytes> [socket_path]\n"
               "  target_gpu:   GPU device ID to create the pool on\n"
               "  bytes:        Size of the buffer to allocate\n"
               "  socket_path:  Unix socket path (default: %s)\n",
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

  int const target_gpu    = std::atoi(argv[1]);
  std::size_t const bytes = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
  char const* socket_path = (argc > 3) ? argv[3] : default_socket_path;

  // Set up socket server
  int server_fd = make_server_socket(socket_path);
  std::fprintf(stderr, "[exporter] Listening on %s\n", socket_path);

  // Set target GPU and warm up context
  RMM_CUDA_TRY(cudaSetDevice(target_gpu));
  RMM_CUDA_TRY(cudaFree(nullptr));  // Context warmup

  // Create IPC-enabled memory resource with POSIX file descriptor handle type
  using handle_type = rmm::mr::cuda_async_memory_resource::allocation_handle_type;
  rmm::mr::cuda_async_memory_resource mr{
    bytes,                              // Initial pool size
    std::nullopt,                       // Release threshold (default)
    handle_type::posix_file_descriptor  // Enable IPC via POSIX FD
  };

  std::fprintf(stderr, "[exporter] Created IPC-enabled memory pool on GPU %d\n", target_gpu);

  // Create a stream for async operations
  rmm::cuda_stream stream{};

  // Allocate a buffer from the pool
  // Note: We use allocate() directly from the memory resource since device_uvector
  // would manage lifetime, but we need the raw pointer for IPC export
  void* device_ptr = mr.allocate(stream.view(), bytes);
  stream.synchronize();

  std::fprintf(stderr, "[exporter] Allocated %zu bytes at %p\n", bytes, device_ptr);

  // Export the pool as a shareable handle (POSIX file descriptor)
  cudaMemPool_t pool_handle = mr.pool_handle();
  int pool_fd               = -1;
  RMM_CUDA_TRY(cudaMemPoolExportToShareableHandle(
    &pool_fd, pool_handle, cudaMemHandleTypePosixFileDescriptor, 0));

  // Export the pointer (allocation) metadata
  cudaMemPoolPtrExportData export_data{};
  RMM_CUDA_TRY(cudaMemPoolExportPointer(&export_data, device_ptr));

  std::fprintf(stderr, "[exporter] Exported pool FD=%d and pointer metadata\n", pool_fd);

  // Accept importer connection
  int client_fd = accept_client(server_fd);
  std::fprintf(stderr, "[exporter] Client connected\n");

  // Send metadata to importer:
  // 1. Target GPU ID
  // 2. Buffer size in bytes
  // 3. Pool file descriptor (via SCM_RIGHTS)
  // 4. Pointer export data
  send_all(client_fd, &target_gpu, sizeof(target_gpu));
  send_all(client_fd, &bytes, sizeof(bytes));
  send_fd(client_fd, pool_fd);
  send_all(client_fd, &export_data, sizeof(export_data));

  std::fprintf(stderr, "[exporter] Sent pool FD and export data to importer\n");

  // Wait for importer to signal completion
  char done_signal = 0;
  recv_all(client_fd, &done_signal, sizeof(done_signal));
  std::fprintf(stderr, "[exporter] Importer finished (signal=%c)\n", done_signal);

  // Cleanup
  ::close(client_fd);
  ::close(server_fd);
  ::close(pool_fd);
  ::unlink(socket_path);

  // Deallocate buffer (must be done before memory resource is destroyed)
  mr.deallocate(stream.view(), device_ptr, bytes);
  stream.synchronize();

  std::fprintf(stderr, "[exporter] Cleanup complete\n");

  return 0;
}
