# Device Symmetric Collectives

## Overview
This directory implements GPU-side symmetric collective operations that use direct peer-to-peer memory access via the Low-Latency Shared Address space (LSA). These kernels are optimized for modern GPU interconnects that support symmetric memory access patterns, enabling all ranks to directly read/write each other's memory without host involvement.

## Files

### kernel.cuh
**Purpose**: Declares the entry point functions for all symmetric collective kernels.

**Key Functions**:
- `ncclSymkRun_AllReduce_AGxLL_R()`: AllReduce using AllGather-style Low-Latency protocol with reduction
- `ncclSymkRun_AllReduce_AGxLLMC_R()`: AllReduce with multimem (NVLS) support
- `ncclSymkRun_AllReduce_RSxLD_AGxST()`: AllReduce using ReduceScatter then AllGather (split algorithm)
- `ncclSymkRun_AllReduce_RSxLDMC_AGxSTMC()`: Split AllReduce with multimem support
- `ncclSymkRun_AllGather_LL/LLMC()`: AllGather using Low-Latency protocol
- `ncclSymkRun_AllGather_ST/STMC()`: AllGather using simple Store protocol
- `ncclSymkRun_ReduceScatter_LL/LD/LDMC()`: ReduceScatter variants (Low-Latency, Load, Load-Multimem)

### primitives.cuh
**Purpose**: Provides helper primitives and utilities for symmetric kernels.

**Key Structures**:
- `ncclSymkArgsHandler`: Wraps kernel arguments, provides work partitioning and iteration helpers
  - `forEachWork()`: Iterates over multiple work items assigned to a block
  - `singleWork()`: Processes a single work item per block
  - `getWorkRange()`: Computes which portion of work a block should process
  - `getWorkRangeFused()`: Handles fused operations across multiple blocks

**Key Functions**:
- `flattenIx()`: Flattens multi-dimensional thread indices to linear index for work distribution
- `ncclSymkAccumType`: Type trait selecting accumulator type (e.g., float for half precision)

**Important Concepts**:
- `NCCL_GRID_CONSTANT`: Grid-constant annotation for Volta+ GPUs
- Work is partitioned into cells (NCCL_SYM_KERNEL_CELL_SIZE) for load balancing
- Supports fractional work distribution across blocks using 16-bit fixed-point arithmetic

### all_reduce.cuh
**Purpose**: Implements AllReduce collective operations.

**Key Functions**:
- `allreduceDeep()`: Core reduction loop for aligned bulk data
  - Reads local data, reduces with all peers' data, broadcasts result
  - Template parameterized by pack size (1-16 bytes), unroll factors
  - Uses warp-level parallelism with configurable unrolling
- `allreduceEnds()`: Handles prefix/suffix bytes that don't fit bulk alignment
- `allreduce()`: Top-level coordinator, selects alignment strategy
- `allreduceMultimem()`: Variant using NVLS multimem for hardware acceleration
- `ncclSymkRun_AllReduce_RSxLD_AGxST()`: Split algorithm (ReduceScatter + AllGather)
- `ncclSymkRun_AllReduce_AGxLL_R()`: AllGather-style using LLA2A (Low-Latency All-to-All)

**Algorithm Details**:
1. **Deep Path**: Processes bulk-aligned data in chunks divisible by nRanks*nBlocks
2. **Reduction Phase**: Each thread loads from all peers using `peerPtr()`, accumulates using reduction operator
3. **Broadcast Phase**: Writes reduced result to all peers' output buffers
4. **Type Casting**: Supports mixed-precision (e.g., FP16 input, FP32 accumulation)

### reduce_scatter.cuh
**Purpose**: Implements ReduceScatter collective operations.

**Key Functions**:
- `reduceDeep()`: Core reduction loop, similar to allreduce but writes only to local output
- `reduceEnds()`: Handles misaligned prefix/suffix elements
- `reduce()`: Top-level function selecting alignment path
- `reduceMultimem()`: NVLS-accelerated variant using multimem loads
- `ncclSymkRun_ReduceScatter_LD()`: Load-based ReduceScatter
- `ncclSymkRun_ReduceScatter_LDMC()`: Load-based with multimem
- `ncclSymkRun_ReduceScatter_LL()`: Low-latency variant using LLA2A
- `ncclSymkRun_ReduceScatter_LL_body()`: Core LL algorithm with pack-based sends/receives

**Algorithm**:
- Each rank reduces a different slice of the input array
- Rank r processes elements [r*nElts, (r+1)*nElts)
- Reads from all peers' input slices, reduces locally
- Only the local rank's portion of the result is written

### all_gather.cuh
**Purpose**: Implements AllGather collective operations.

**Key Functions**:
- `bcastDeep()`: Core broadcast loop for bulk-aligned data
- `bcastEnds()`: Handles prefix/suffix bytes
- `bcast()`: Top-level coordinator selecting alignment strategy
- `bcastMultimem()`: NVLS-accelerated broadcast
- `ncclSymkRun_AllGather_ST()`: Store-based AllGather
- `ncclSymkRun_AllGather_STMC()`: Store-based with multimem
- `ncclSymkRun_AllGather_LL()`: Low-latency variant using LLA2A
- `allgather_LL_body()`: Core LL algorithm with broadcast/receive pattern

**Algorithm**:
- Each rank broadcasts its local data to all peers
- Rank r writes to output offset [r*nElts, (r+1)*nElts) across all peers
- Uses direct stores to peer memory or LLA2A for coordination
- In-place operation skips local copy

## Key Concepts

### Symmetric Memory Access
- All ranks have direct access to each other's memory via shared address space
- `ncclSymPtr<T>` abstracts pointer conversion:
  - `localPtr()`: Access own memory
  - `peerPtr(team, rank)`: Access peer's memory
  - `lsaPtr(rank)`: Access via LSA
  - `multimemPtr(handle)`: Access via NVLS multimem

### Low-Latency All-to-All (LLA2A)
- Coordination mechanism for symmetric kernels
- Provides `bcast()`, `send()`, `recv()`, `recvReduce()` operations
- Epochs separate batches of communication for pipelining
- Supports unrolled receives for better instruction-level parallelism

### Multimem (NVLS)
- Hardware-accelerated multicast memory operations
- Enables atomic reduction across multiple memories
- Special load/store instructions (`applyLoadMultimem`, `multimem_st_global`)
- Used with "_MC" suffix variants of kernels

### Alignment Optimization
- Detects common alignment between input/output buffers
- Selects largest possible pack size (16, 4 bytes) for bulk transfers
- Falls back to element-wise operations for misaligned prefix/suffix
- Alignment requirements vary by algorithm (16B for deep paths, 8B for LL)

### Work Distribution
- Work partitioned into cells for fine-grained load balancing
- Blocks assigned fractional work ranges using 16-bit fixed-point
- Thread numbering schemes vary by algorithm:
  - Round-robin by rank then block for split algorithms
  - Round-robin by block for simple algorithms

### Reduction Operators
- Template parameterized by reduction function (Sum, Min, Max, etc.)
- Accumulator type may differ from data type for precision
- `applyReduce()` applies the reduction operator
- `applyCast()` converts between data and accumulator types

## Dependencies
- `sym_kernels.h`: Shared definitions for symmetric kernels
- `nccl_device.h`: Device-side NCCL primitives
- `../op128.h`: 128-bit memory operations
- `../reduce_kernel.h`: Reduction operator implementations
- `bitops.h`: Bit manipulation utilities
- `collectives.h`: General collective definitions

## Integration
Symmetric kernels are invoked by the NCCL scheduler when:
1. All ranks have established symmetric memory mappings
2. The communication pattern fits a symmetric algorithm
3. Hardware supports required features (P2P, NVLS)

The scheduler:
- Partitions work across multiple kernel launches (channels)
- Provides `ncclSymkDevWorkArgs` containing work descriptors
- Coordinates synchronization via barriers
- Selects algorithm variant based on data size and alignment

These kernels offer the highest performance path in NCCL by:
- Eliminating CPU involvement during data transfer
- Maximizing memory bandwidth with direct peer access
- Leveraging hardware acceleration (NVLS) when available
- Minimizing synchronization overhead with fine-grained work distribution
