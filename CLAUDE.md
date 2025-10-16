# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NCCL (NVIDIA Collective Communications Library) is a stand-alone library providing optimized GPU communication primitives for inter-GPU communication. It implements all-reduce, all-gather, reduce, broadcast, reduce-scatter, and send/receive operations optimized for PCIe, NVLink, NVswitch, and networking (InfiniBand/TCP).

## Build Systems

NCCL supports both Make and CMake build systems.

### Make Build (Primary)
```bash
# Basic build
make -j src.build

# With custom CUDA path
make src.build CUDA_HOME=/path/to/cuda

# With specific GPU architectures (faster compilation)
make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"

# Debug build
make -j src.build DEBUG=1

# With sanitizers
make -j src.build ASAN=1      # Address sanitizer
make -j src.build UBSAN=1     # Undefined behavior sanitizer

# Enable tracing
make -j src.build TRACE=1

# Treat warnings as errors
make -j src.build WERROR=1

# Build with RDMA support
make -j src.build RDMA_CORE=1 MLX5DV=1
```

### CMake Build (Alternative)
```bash
# Basic build
mkdir build && cd build
cmake ..
cmake --build . -j

# With options
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
         -DDEBUG=ON \
         -DASAN=ON \
         -DTRACE=ON \
         -DRDMA_CORE=ON \
         -DMLX5DV=ON
cmake --build . -j
```

Build output goes to `build/` directory by default (configurable with `BUILDDIR`).

### Build Examples
```bash
# Build examples (requires NCCL to be built first)
make -j examples

# Build with MPI support
make -j examples MPI=1

# Or from examples directory
cd examples
make NCCL_HOME=/path/to/nccl/build [MPI=1]
```

### Install Packages
```bash
# Debian/Ubuntu
make pkg.debian.build    # Creates .deb in build/pkg/deb/

# RedHat/CentOS
make pkg.redhat.build    # Creates .rpm in build/pkg/rpm/

# OS-agnostic tarball
make pkg.txz.build       # Creates .tar.xz in build/pkg/txz/
```

## Architecture

### Core Components

**Communication Primitives** (`src/collectives.cc`, `src/enqueue.cc`):
- Main entry points for collective operations (AllReduce, Broadcast, ReduceScatter, etc.)
- `enqueue.cc` handles work submission to GPU channels and kernel launches
- Supports both in-place and out-of-place operations

**Initialization** (`src/init.cc`):
- Communicator creation and configuration
- GPU topology detection and channel setup
- Integration with NVML, CUDA runtime, and network transports

**Transport Layer** (`src/transport.cc`, `src/transport/`):
- Abstracts different interconnect types (P2P, SHM, NET)
- Handles data movement between GPUs across different topologies
- Network transport plugins for InfiniBand, TCP/IP, custom networks

**Channel Management** (`src/channel.cc`):
- Work channels for parallel communication execution
- Ring and tree algorithm implementations
- Connection setup between peers

**Proxy Threads** (`src/proxy.cc`):
- CPU-side threads managing network operations
- Offloads network communication from GPU
- Handles asynchronous progress

**Device Code** (`src/device/`):
- CUDA kernels implementing collective algorithms
- Generated code for different data types and operations (`generate.py`)
- Symmetric memory support (`src/device/symmetric/`)

**Bootstrap** (`src/bootstrap.cc`):
- Initial rank-to-rank connection establishment
- Out-of-band communication for setup phase
- Supports multiple bootstrap mechanisms

**Graph Support** (`src/graph/`):
- CUDA graph capture and replay support
- Optimizations for static communication patterns

**Scheduler** (`src/scheduler/`):
- Work scheduling and channel assignment
- Optimizes communication patterns

**Memory Management** (`src/allocator.cc`, `src/register/`):
- Memory allocation strategies
- Buffer registration for RDMA
- Symmetric memory windows (added in 2.27)

### Plugin System

**Network Plugins** (`src/plugin/net/`):
- External network transport plugins
- Custom interconnect support
- Configurable via `NCCL_NET_MAX_PLUGINS`

**Profiler Plugins** (`src/plugin/profiler/`, `ext-profiler/`):
- Performance profiling hooks
- Network profiling support (`NET_PROFILER=1`)
- Custom profiler implementations

**Tuner Plugins** (`src/plugin/tuner/`, `ext-tuner/`):
- Algorithm and parameter auto-tuning
- Custom optimization strategies

### Reliability and Monitoring

**RAS (Reliability, Availability, Serviceability)** (`src/ras/`):
- Error detection and reporting
- Health monitoring
- Client daemon (`ncclras` binary from `ras/client.cc`)

## Key Configuration

### Build-Time Options (Make)
- `DEBUG`: Enable debug symbols and disable optimizations
- `TRACE`: Enable detailed tracing output
- `ASAN/UBSAN`: Enable sanitizers (run with `ASAN_OPTIONS=protect_shadow_gap=0`)
- `PROFAPI`: Enable profiling API (default ON)
- `NVTX`: Enable NVTX markers (default ON)
- `RDMA_CORE`: Enable RDMA verbs support
- `MLX5DV`: Enable Mellanox direct verbs
- `NET_PROFILER`: Enable network profiling
- `NVCC_GENCODE`: Control target GPU architectures
- `CUDA_HOME`: CUDA installation path (default `/usr/local/cuda`)
- `BUILDDIR`: Output directory (default `./build`)

### Version Management
Version information is stored in `makefiles/version.mk` with:
- `NCCL_MAJOR`, `NCCL_MINOR`, `NCCL_PATCH`, `NCCL_SUFFIX`, `PKG_REVISION`

## Testing

Official NCCL tests are maintained in a separate repository:
```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make
./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```

## Important Conventions

### CUDA Architecture Support
- Automatically determined based on CUDA version
- CUDA 8: SM 60,61
- CUDA 9+: adds SM 70 (Volta)
- CUDA 11+: adds SM 80 (Ampere)
- CUDA 11.8+: adds SM 90 (Hopper)
- CUDA 12.8+: adds SM 100,120 (Blackwell)
- CUDA 13+: SM 110, requires C++17

### Code Standards
- C++14 minimum (C++17 for CUDA 13+)
- CUDA kernels use extended lambda (`--expt-extended-lambda`)
- Max register count: 96 per thread
- Error checking via `ncclResult_t` return codes

### Public API
- Main header: `src/nccl.h.in` (generated to `nccl.h`)
- Device API: `nccl_device.h` and `include/nccl_device/*.h`
- Visibility controlled via `-fvisibility=hidden` and `__attribute__((visibility("default")))`
- `PROFAPI` build adds `p`-prefixed function aliases for profiling

### Plugin Development
Example plugins provided in:
- `ext-net/example/` - Network transport plugin
- `ext-profiler/example/` - Profiler plugin
- `ext-tuner/example/` - Tuner plugin

## Documentation

- Official docs: https://docs.nvidia.com/deeplearning/nccl/user-guide/
- Examples with detailed READMEs in `examples/` directory
- Each example progresses from basic to advanced usage
