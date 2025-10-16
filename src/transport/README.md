# Transport Layer

## Overview
The transport directory implements the communication mechanisms NCCL uses to move data between GPUs. It provides multiple transport types optimized for different connectivity scenarios (intra-node, inter-node, direct GPU, proxied).

## Files

### p2p.cc
**Purpose**: Point-to-point transport for GPU-to-GPU communication.

**Transport Types** (enum p2pType):
- `P2P_DIRECT`: Direct GPU memory access via CUDA IPC or peer access
- `P2P_INTERMEDIATE`: Indirect transfer through intermediate GPU
- `P2P_IPC`: Inter-process communication via shared memory
- `P2P_CUMEM`: CUDA Unified Memory based transfers

**Key Structures**:
- `ncclP2pBuff`: Buffer descriptor (pointer, size, IPC handle)
- `p2pConnectInfo`: Connection setup information exchanged during initialization
- `p2pResources`: Per-connection state (memory pointers, IPC handles, proxy info)
- `p2pShmProxyInfo`: Shared memory info for copy engine (CE) mediated transfers

**Key Capabilities**:
- Zero-copy transfers when GPUs have direct access
- Fallback to CPU-mediated copies when direct access unavailable
- Copy engine support for peer-to-peer without CUDA peer access
- Support for CUDA memory handle-based transfers

### shm.cc
**Purpose**: Shared memory transport for intra-node communication.

Implements:
- POSIX shared memory for control structures
- Host-to-device and device-to-host staging
- Producer-consumer FIFOs for data movement
- Synchronization via atomic operations

Used when:
- GPUs on same node without direct P2P
- As fallback when P2P registration fails
- For communication with proxy threads

### net.cc
**Purpose**: Network transport for inter-node communication.

Integrates with network plugins to:
- Send/receive data across nodes
- Manage network buffer registration
- Handle proxy thread coordination
- Support multiple network interfaces (rails)

**Key Operations**:
- `ncclNetConnect()`: Establish inter-node connections
- `ncclNetRegBuffer()`: Register buffers with network hardware
- `ncclNetSend/Recv()`: Enqueue network operations
- Progress tracking via proxy threads

### net_ib.cc
**Purpose**: InfiniBand/RoCE specific optimizations.

Implements:
- RDMA queue pair management
- Adaptive routing support
- GPUDirect RDMA for zero-copy transfers
- Hardware transport configuration

### net_socket.cc
**Purpose**: TCP/IP socket-based network transport.

Fallback network implementation:
- Works on any IP network
- No special hardware requirements
- Lower performance than RDMA
- Used for bootstrap and control plane

### nvls.cc
**Purpose**: NVLink Switch (NVLS) transport.

Leverages NVLS/NVSwitch for:
- High-bandwidth intra-node communication
- Multicast operations across GPUs
- Symmetric memory access patterns
- Hardware-accelerated reductions

Provides:
- NVLS registration and memory mapping
- Multicast handle management
- Integration with symmetric kernels

### coll_net.cc
**Purpose**: Collective network offload transport.

Supports switch-based collective acceleration:
- In-network reduction operations
- Broadcast/multicast in hardware
- Lower latency for supported collectives
- Integration with Sharp (Scalable Hierarchical Aggregation Protocol)

### profiler.cc
**Purpose**: Transport profiling and instrumentation.

Hooks for:
- Bandwidth measurement
- Latency tracking
- Operation counting
- Performance analysis

### generic.cc
**Purpose**: Generic transport operations and utilities.

Common functionality:
- Transport selection logic
- Capability negotiation
- Setup and teardown
- Error handling

## Key Concepts

### Transport Selection
NCCL chooses transport based on:
1. **Topology**: Physical connectivity between ranks
2. **Capabilities**: Available hardware features (P2P, RDMA, NVLS)
3. **Buffer Location**: Device, host, managed memory
4. **Size**: Small vs. large transfers may use different paths
5. **Algorithm**: Some collectives prefer specific transports

Selection hierarchy (fastest to slowest):
1. NVLS multicast (intra-node, symmetric)
2. Direct P2P (intra-node, peer access)
3. Copy Engine P2P (intra-node, no peer access)
4. GPUDirect RDMA (inter-node, registered buffers)
5. Shared memory + proxy (intra-node fallback)
6. Network + proxy (inter-node standard)
7. Socket (fallback)

### Connection Establishment
Three-phase process:
1. **Setup**: Exchange capabilities and buffer information
2. **Connect**: Establish communication channels (QPs, memory mappings)
3. **Test**: Verify connection integrity before use

Information exchanged in `p2pConnectInfo`:
- Rank identifiers
- Buffer addresses and sizes
- IPC handles for shared memory/CUDA IPC
- Transport-specific descriptors

### Proxy Architecture
Some transports require CPU proxy threads:
- **Network Proxy**: Issues network operations on behalf of GPU
- **Service Proxy**: Handles asynchronous operations
- **CE Proxy**: Manages copy engine operations

Proxies communicate with GPU via shared memory:
- GPU writes operation descriptors to shared FIFO
- Proxy reads descriptors and executes operations
- Completion signaled via shared memory flags

### Buffer Registration
Critical for high-performance transfers:
- **Network**: Pin memory and register with NIC
- **NVLS**: Map buffers into multicast address space
- **P2P**: Export/import CUDA IPC handles
- **IPC**: Create shared memory mappings

Registration is cached to amortize overhead across operations.

### Symmetric vs. Asymmetric Paths
- **Symmetric**: All ranks access shared memory directly (NVLS, P2P)
- **Asymmetric**: Different roles (sender/receiver) with different code paths (NET, SHMS)

Symmetric paths enable simpler, faster kernels but require specific hardware support.

### Multi-Rail Support
Multiple network interfaces used simultaneously:
- Each rail has independent send/receive queues
- Load balanced across rails
- Aggregated bandwidth
- Fault tolerance if rail fails

## Dependencies
- `graph.h`: Topology graph for transport selection
- `net.h`: Network plugin interface
- `register.h`: Buffer registration system
- `shm.h`: Shared memory utilities
- `comm.h`: Communicator state

## Integration

### Initialization Flow
1. Topology discovery builds connectivity graph
2. Graph algorithms compute optimal paths
3. Transport selection assigns transport to each connection
4. Setup phase exchanges connection info
5. Connect phase establishes channels
6. Resources allocated and registered

### Operation Flow
1. Collective scheduler assigns work to channels
2. Channels use pre-established transport connections
3. GPU kernels write data via transport
4. Proxies progress network operations (if needed)
5. Completion detected via polling or events
6. Synchronization ensures operation finishes before next

### Cleanup Flow
1. Flush in-flight operations
2. Unregister buffers
3. Close connections (QPs, sockets)
4. Free resources
5. Destroy proxy threads

## Performance Considerations

### Latency
Transport latency hierarchy:
- P2P: ~1-2 microseconds
- NVLS: ~1-2 microseconds
- GPUDirect RDMA: ~5-10 microseconds
- Standard RDMA: ~10-20 microseconds
- Shared memory: ~5-10 microseconds
- Socket: ~100+ microseconds

### Bandwidth
Theoretical maximums:
- NVLink (per link): 25-112 GB/s depending on generation
- PCIe Gen3 x16: ~12 GB/s
- PCIe Gen4 x16: ~24 GB/s
- PCIe Gen5 x16: ~48 GB/s
- InfiniBand HDR: ~25 GB/s per port
- InfiniBand NDR: ~50 GB/s per port
- Ethernet 100G: ~12.5 GB/s

### Protocol Overhead
Sources of overhead:
- Connection setup: ~1-10 milliseconds
- Buffer registration: ~100-1000 microseconds
- Proxy wakeup: ~1-5 microseconds
- Completion polling: ~100-500 nanoseconds

The transport layer is the foundation of NCCL's performance, carefully orchestrating data movement across the complex memory and network hierarchies of modern GPU clusters.
