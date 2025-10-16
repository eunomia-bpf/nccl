# Buffer Registration System

## Overview
This directory implements NCCL's buffer registration and caching mechanism. Registration pins memory and establishes transport-specific handles (network, NVLS, IPC) to enable zero-copy data transfers without repeated setup overhead.

## Files

### register.cc
**Purpose**: Core registration cache management and lifecycle functions.

**Key Functions**:
- `ncclRegister()`: Registers a buffer, adding it to the cache or incrementing refcount
  - Aligns buffer to page boundaries
  - Maintains sorted cache for efficient lookup
  - Supports separate refcounts for graph vs. local usage
- `regCleanup()`: Deregisters buffer across all transport types (NET, NVLS, COLLNET, IPC)
- `ncclRegLocalIsValid()`: Checks if registration is valid for local use

**Registration States** (tracked in `reg->state`):
- `NET_REG_COMPLETE`: Network transport registration finished
- `NVLS_REG_COMPLETE`: NVLS multicast registration finished
- `COLLNET_REG_COMPLETE`: Collective network registration finished
- `IPC_REG_COMPLETE`: Inter-process communication registration finished

**Cache Structure**:
- **Sorted Array**: Registrations sorted by address for binary search
- **Refcounting**: Tracks graph references and local references separately
- **Handle Storage**: Maintains transport-specific handles per registration

### coll_reg.cc
**Purpose**: Handles registration for collective operations (likely includes network collective operations).

### sendrecv_reg.cc
**Purpose**: Handles registration for point-to-point send/receive operations.

## Key Concepts

### Registration Cache
The cache prevents redundant registration overhead:
1. **Lookup**: Check if buffer range already covered by existing registration
2. **Reuse**: Increment refcount if found
3. **Register**: Pin memory and create transport handles if new
4. **Cleanup**: Deregister when refcount reaches zero

### Page Alignment
Buffers are expanded to page boundaries:
- `begAddr = buffer_start & -pageSize` (round down)
- `endAddr = (buffer_end + pageSize - 1) & -pageSize` (round up)

This ensures:
- Registrations cover entire pages
- Multiple buffers on same page share registration
- Alignment requirements met for DMA operations

### Multi-Transport Support
A single buffer may be registered across multiple transports simultaneously:
- **Network (NET)**: For inter-node communication via InfiniBand/RoCE
- **NVLS**: For NVLink multicast within a node
- **CollNet**: For switch-based collective offload
- **IPC**: For shared memory access between local processes

Each transport maintains its own handle in the registration structure.

### Reference Counting
Two independent refcount types:
- **Graph References**: Buffers registered by CUDA graph capture
- **Local References**: Buffers registered by direct API calls

This separation allows different lifetimes for graph-captured vs. manually registered buffers.

### Cleanup Process
When refcount reaches zero:
1. Iterate through all transport registrations
2. Call transport-specific deregister function
3. Free associated memory structures
4. Remove from cache

Partial failures are logged but don't block cleanup of other transports.

## Dependencies
- `comm.h`: Communicator structure containing registration cache
- `net.h`: Network transport registration/deregistration
- `transport.h`: Transport abstraction layer
- NVLS, CollNet, IPC subsystems for transport-specific operations

## Integration
Buffer registration integrates with:
1. **Collective Operations**: Auto-register user buffers before collective
2. **CUDA Graphs**: Register buffers during graph capture for replay efficiency
3. **Proxy Threads**: Provide registered handles to network proxies
4. **Memory Management**: Track pinned memory limits and usage

### Performance Impact
Registration is expensive (system call, page locking, handle creation) but amortized:
- **Cold Path**: First use incurs full registration cost (~10-100 microseconds)
- **Hot Path**: Cache hits have negligible overhead (~10-100 nanoseconds)
- **Trade-off**: Cache memory usage vs. registration overhead

The registration cache is critical for achieving high performance in iterative workloads where the same buffers are used repeatedly.
