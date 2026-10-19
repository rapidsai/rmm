# IPC Pool Sharing Example

This example demonstrates CUDA IPC (Inter-Process Communication) memory pool
sharing using RMM's `cuda_async_memory_resource`. It shows how to share a
GPU memory pool between processes and perform multi-GPU peer-to-peer writes
to the shared allocation.

## What This Example Demonstrates

1. Creating an IPC-enabled memory pool with `cuda_async_memory_resource`
2. Exporting the pool handle via POSIX file descriptor
3. Importing the pool in another process
4. Wrapping an imported pool with `cuda_async_view_memory_resource`
5. Multi-GPU peer-to-peer writes to a shared allocation

## Components

### pool_exporter

Creates a shareable memory pool on a target GPU and exports it to the importer:
- Uses `cuda_async_memory_resource` with `allocation_handle_type::posix_file_descriptor`
- Allocates a buffer from the pool
- Exports pool handle via `cudaMemPoolExportToShareableHandle()`
- Exports pointer metadata via `cudaMemPoolExportPointer()`
- Sends data to importer via Unix domain socket with SCM_RIGHTS

### pool_importer

Imports the shared pool and performs multi-GPU writes:
- Receives pool handle and pointer metadata via Unix domain socket
- Imports pool via `cudaMemPoolImportFromShareableHandle()`
- Imports pointer via `cudaMemPoolImportPointer()`
- Wraps imported pool with `cuda_async_view_memory_resource`
- Spawns threads that write to disjoint regions from different GPUs using P2P

## Compile and Execute

```bash
# Configure project
cmake -S . -B build/

# Build
cmake --build build/
```

### Running the Example

Open two terminals. In the first terminal, start the exporter:

```bash
# Syntax: pool_exporter <target_gpu> <bytes> [socket_path]
# Example: Create 256MB buffer on GPU 0
build/pool_exporter 0 268435456
```

In the second terminal, start the importer:

```bash
# Syntax: pool_importer <target_gpu> <writer_gpu_0> [writer_gpu_1...] [socket_path]
# Example: Import on GPU 0, write from GPUs 1, 2, 3
build/pool_importer 0 1 2 3
```

For single-GPU systems or testing:

```bash
# Terminal 1: Export from GPU 0
build/pool_exporter 0 67108864

# Terminal 2: Import and write from GPU 0
build/pool_importer 0 0
```

## Requirements

- Linux with POSIX file descriptor support
- Multi-GPU system recommended for P2P demonstration
- GPUs must support peer access for P2P writes

## RMM APIs Demonstrated

- `rmm::mr::cuda_async_memory_resource` - IPC-enabled memory pool
- `rmm::mr::cuda_async_view_memory_resource` - Non-owning view of imported pool
- `rmm::cuda_stream` - RAII stream management
- `rmm::device_uvector` - Device memory container
- `RMM_CUDA_TRY` / `RMM_EXPECTS` - Error handling macros
