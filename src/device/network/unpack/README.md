# Network Device Unpack

## Overview
This directory implements GPU-side unpacking functionality for network receive operations. It handles copying data from internal bounce buffers to user buffers on the device, optimizing for alignment and using warp-level parallelism.

## Files

### unpack_defs.h
**Purpose**: Defines data structures and constants for network device unpacking.

**Key Structures**:
- `loadMeta`: 16-byte aligned structure containing metadata for a single memory copy operation
  - `src_off`: Source offset in bounce buffer
  - `len`: Length of data to copy
  - `dst_off`: Destination offset in user buffer
- `netUnpackMeta`: Per-request metadata array containing up to MAX_QUEUE_DEPTH requests, each with up to MAX_SLICE_PAGES page descriptors
- `unpackNetDeviceHandle`: Device handle containing pointers to metadata, bounce buffer, and queue head position
- `unpackShmem`: Shared memory structure for bounce buffer pointer
- `unpackGroupShmem`: Per-group shared memory containing heads, metadata pointers, and peer masks

**Key Constants**:
- `NCCL_NET_DEVICE_UNPACK_MAX_QUEUE_DEPTH`: 16 (maximum number of in-flight requests)
- `NET_UNPACK_MAX_SLICE_SIZE`: 4MB per receive call
- `SLICE_PAGE_SIZE`: 4096 bytes
- `WARP_SHM_PAGE_CNT`: 4 (pages of metadata per warp in shared memory)

### unpack.h
**Purpose**: Implements the core unpacking algorithms executed on GPU.

**Key Functions**:
- `ncclNetDeviceUnpackSetup()`: Maps handles to group and peer indices, initializes shared memory pointers
- `ncclNetDeviceIncrementHead()`: Advances the queue head for a peer
- `ncclNetDeviceSaveHead()`: Saves the current head position back to the handle
- `bulkLoad<sz>()`: Template functions for aligned bulk data copying (1, 2, 4, 8, 16 byte alignments)
- `ncclNetDeviceUnpack<Recv>()`: Main entry point, dispatches to unpacking for each peer with pending receives
- `ncclNetDeviceUnpackInner()`: Core unpacking logic that processes metadata and copies data
- `load64gpu()`: GPU memory load with relaxed consistency (Volta+) or volatile loads
- `ppw()`: Calculate pages-per-warp for workload distribution

**Important Algorithms**:
1. **Alignment-based bulk copying**: Detects common alignment between source and destination to use wider loads/stores (16B down to 1B)
2. **Warp-level parallelism**: Distributes page metadata across warps, each warp processes multiple pages
3. **Two-stage loading**: First loads metadata from global to shared memory, then processes data copies
4. **Queue management**: Circular buffer with head pointers tracks in-flight requests

## Key Concepts

### Data Flow
1. Network plugin writes received data to bounce buffer and populates metadata array
2. GPU threads load metadata describing which portions of bounce buffer map to which user buffer offsets
3. Threads cooperatively copy data from bounce buffer to final destination using optimal alignment
4. Queue head advances to indicate completion

### Memory Hierarchy
- **Global Memory**: netUnpackMeta structure contains per-request page descriptors
- **Shared Memory**: Each warp caches up to WARP_SHM_PAGE_CNT page descriptors to reduce global memory traffic
- **Registers**: BytePack registers hold 16 bytes during bulk transfers

### Optimization Techniques
- **Alignment detection**: Uses bitwise operations to find common alignment and select widest possible loads/stores
- **Warp synchronization**: Strategic `__syncwarp()` calls ensure metadata is loaded before processing
- **Volatile loads**: Ensures visibility of data written by network plugin without heavy fencing

## Dependencies
- `op128.h`: 128-bit load/store operations
- `bitops.h`: Bit manipulation utilities
- `device.h`: Device-side common definitions
- `common.h`: Shared constants and macros
- `ncclShmem`: Global shared memory structure containing group and plugin state

## Integration
This component is part of the network receive path in NCCL's device-side execution:
1. Invoked during collective operations when data arrives from network peers
2. Works in conjunction with the network plugin's device-side interface
3. Operates on a per-group, per-peer basis allowing multiple concurrent receives
4. Integrates with NCCL's work scheduler which provides thread allocation and synchronization

The unpacking system bridges the gap between the network plugin's internal buffer representation (potentially scattered/segmented) and NCCL's requirement for data in user-specified buffers, performing this translation efficiently on the GPU without CPU involvement.
