# Miscellaneous Utilities

## Overview
This directory contains utility functions and wrapper libraries for external dependencies. It provides abstraction layers for CUDA, NVML, InfiniBand, networking, and shared memory operations used throughout NCCL.

## Files

### utils.cc
**Purpose**: General utility functions for device identification, host naming, and hashing.

**Key Functions**:
- `ncclCudaCompCap()`: Returns current device compute capability (e.g., 70 for Volta, 80 for Ampere)
- `busIdToInt64()` / `int64ToBusId()`: Convert between PCIe bus ID strings and int64 representation
- `getBusId()`: Get PCIe bus ID for a CUDA device
- `getHostName()`: Retrieve hostname with optional delimiter trimming
- `getHostHash()`: Generate unique host identifier from hostname + boot_id

### cudawrap.cc
**Purpose**: Wrapper for CUDA runtime and driver APIs with dynamic loading support.

Provides versioned API wrappers allowing NCCL to:
- Load CUDA symbols dynamically at runtime
- Support multiple CUDA versions
- Handle missing symbols gracefully
- Avoid hard dependencies on specific CUDA versions

### nvmlwrap.cc
**Purpose**: Wrapper for NVIDIA Management Library (NVML) APIs.

Functions for:
- GPU topology discovery
- Power and thermal monitoring
- Device identification and properties
- Link status and bandwidth queries

### ibvwrap.cc / ibvsymbols.cc
**Purpose**: Wrapper for InfiniBand Verbs library (libibverbs).

Provides abstraction for:
- RDMA queue pair management
- Memory registration for RDMA
- Completion queue handling
- Device and port queries

### mlx5dvwrap.cc / mlx5dvsymbols.cc
**Purpose**: Wrapper for Mellanox device-specific APIs (mlx5 direct verbs).

Enables advanced features:
- GPU Direct RDMA optimizations
- Hardware-accelerated collectives
- Direct data placement

### gdrwrap.cc
**Purpose**: Wrapper for GPUDirect RDMA Copy library.

Facilitates:
- Direct GPU memory to NIC transfers
- Bypassing CPU for RDMA operations
- High-bandwidth inter-node communication

### socket.cc
**Purpose**: Socket utilities for TCP/IP networking.

Functions for:
- Socket creation and connection
- Send/receive with timeout
- Address resolution
- Network interface discovery

### ipcsocket.cc
**Purpose**: Unix domain socket utilities for local IPC.

Used for:
- Communication between NCCL proxies and main threads
- Control plane messaging
- Process coordination on same node

### shmutils.cc
**Purpose**: Shared memory utilities.

Provides:
- POSIX shared memory creation/destruction
- Memory-mapped file I/O
- Inter-process buffer sharing

### param.cc
**Purpose**: Environment variable parameter system.

Implements:
- Centralized parameter registration
- Type-safe environment variable parsing
- Default value management
- Runtime parameter modification

### argcheck.cc
**Purpose**: Argument validation for NCCL API functions.

Validates:
- Communicator validity
- Buffer pointer alignment
- Count and datatype consistency
- Operation and reduction compatibility

### strongstream.cc
**Purpose**: Strong stream ordering utilities for CUDA.

Ensures:
- Correct ordering of CUDA operations
- Dependency management between streams
- Synchronization primitives

## Key Concepts

### Dynamic Library Loading
Many wrappers use `dlopen()`/`dlsym()` to:
- Load libraries at runtime instead of link time
- Support optional features (NVML, InfiniBand, GPUDirect)
- Gracefully degrade when libraries are unavailable
- Avoid version conflicts

### Symbol Versioning
Wrappers maintain multiple symbol versions:
- Support older and newer library versions
- Select appropriate symbols at runtime
- Maintain backward compatibility

### Error Handling
Utility functions follow NCCL conventions:
- Return `ncclResult_t` status codes
- Use `NCCLCHECK()` macro for error propagation
- Log warnings/errors via NCCL's logging system

### Host Identification
Host hash generation ensures:
- Unique identification across bare-metal and containers
- Combines hostname with kernel boot ID
- Deterministic across NCCL processes on same host
- Can be overridden via `NCCL_HOSTID` environment variable

### PCIe Bus ID Format
Bus IDs use compact int64 representation:
- Format: `DDDD:BB:DD.F` (Domain:Bus:Device.Function)
- Packed into 64 bits for efficient comparison
- Used for topology analysis and device matching

## Dependencies
External libraries (dynamically loaded):
- `libcuda.so`: CUDA driver
- `libcudart.so`: CUDA runtime
- `libnvidia-ml.so`: NVML
- `libibverbs.so`: InfiniBand verbs
- `libmlx5.so`: Mellanox device library
- `libgdrapi.so`: GPUDirect RDMA

## Integration
These utilities are used throughout NCCL:
- **Initialization**: Device discovery, topology analysis
- **Transport Setup**: Network registration, buffer preparation
- **Communication**: Socket connections, shared memory IPC
- **Monitoring**: Error detection, health checks
- **Configuration**: Parameter parsing, feature detection

The misc directory serves as NCCL's "standard library," providing portable, reliable abstractions over platform-specific functionality.
