# NCCL Device API Implementation

## Overview
This directory contains the host-side implementation of the NCCL device API - functions that can be called from user code to interact with device-side NCCL features. These functions provide abstractions for team management, barrier synchronization, and low-latency all-to-all communication.

## Files

### core.cc
**Purpose**: Implements core team management functions for organizing ranks into logical communication groups.

**Key Functions**:
- `ncclTeamWorld()`: Returns a team representing all ranks in the communicator
  - nRanks = total communicator size
  - rank = current rank
  - stride = 1 (contiguous ranks)
- `ncclTeamLsa()`: Returns a team for Low-latency Shared Address (LSA) space ranks
  - nRanks = number of ranks with symmetric memory access
  - rank = position within LSA group
  - stride = 1
  - Requires `ncclDevrInitOnce()` initialization
- `ncclTeamRail()`: Returns a team for "rail" groups across LSA boundaries
  - Partitions communicator into groups based on LSA size
  - nRanks = total_ranks / lsa_size
  - rank = world_rank / lsa_size
  - stride = lsa_size (interleaved ranks)
- `ncclTeamRankToWorld()`: Converts team-local rank to global world rank
  - Formula: world_rank = my_world_rank + (team_rank - my_team_rank) * stride
- `ncclTeamRankToLsa()`: Converts team-local rank to LSA-local rank
  - Formula: lsa_rank = my_lsa_rank + (team_rank - my_team_rank) * stride

**Concepts**:
- **Team**: Logical grouping of ranks with specific topology
- **World Team**: All ranks in flat topology
- **LSA Team**: Ranks sharing symmetric memory space
- **Rail Team**: Cross-LSA groups for hierarchical collectives
- Teams abstract rank renumbering and stride patterns for flexible algorithm implementation

### mem_barrier.cc
**Purpose**: Implements host API for LSA barrier synchronization primitives.

**Key Functions**:
- `ncclLsaBarrierCreateRequirement()`: Calculates resource requirements for barrier handle
  - Input: team specification, number of barriers needed
  - Output: buffer size and alignment requirements
  - Buffer size: (3 + nRanks) * nBarriers * sizeof(uint32_t)
    - 3 words per barrier for control state
    - nRanks words per barrier for arrival tracking
  - Populates output handle and requirement structure

**Purpose**:
Barriers synchronize GPU threads across ranks in symmetric memory systems. Each barrier has:
- Arrival counters for each rank
- Generation/epoch counter to distinguish barrier instances
- Control state for coordination

Resource requirements are pre-calculated so memory can be allocated before kernel launch.

### ll_a2a.cc
**Purpose**: Implements host API for Low-Latency All-to-All (LLA2A) communication primitive.

**Key Functions**:
- `ncclLLA2ACalcSlots()`: Calculates number of slots needed for LLA2A buffer
  - Formula: maxElts * divUp(maxEltSize, 8)
  - Slots hold 8-byte chunks of data
  - Must accommodate largest expected transfer
- `ncclLLA2ACreateRequirement()`: Calculates resource requirements for LLA2A handle
  - Input: number of blocks, number of slots
  - Output: buffer size and alignment requirements
  - Buffer size: nBlocks * (1 + 2*nSlots) * 16 bytes
    - 1 control word per block
    - 2 buffers (send/recv) with nSlots entries each
    - 16-byte alignment for coalesced access

**Purpose**:
LLA2A enables efficient small-message all-to-all communication patterns in symmetric kernels. It provides:
- Slot-based buffering for concurrent sends/receives
- Double-buffering to overlap communication and computation
- Per-block state to support concurrent kernel blocks

## Key Concepts

### Team Abstraction
Teams provide a flexible way to organize ranks:
- **Stride**: Distance between consecutive ranks in world space
- **Rank Mapping**: Convert between team-local and global ranks
- **Topology Awareness**: Teams encode physical topology (LSA groups, rails)

Example with 8 ranks, lsaSize=4:
- World: ranks [0,1,2,3,4,5,6,7], stride=1
- LSA: ranks [0,1,2,3] on node 0, [0,1,2,3] on node 1, stride=1
- Rail: ranks [0,4] (rail 0), [1,5] (rail 1), [2,6] (rail 2), [3,7] (rail 3), stride=4

### Device Resource Requirements
All device primitives follow a two-phase creation pattern:
1. **Requirement Phase**: Calculate buffer size/alignment without allocation
2. **Allocation Phase**: User allocates memory, NCCL initializes handle

This design:
- Allows user control over memory allocation
- Enables memory pooling and reuse
- Separates resource planning from resource consumption

### Deferred Initialization
Functions accessing device state call `ncclDevrInitOnce()`:
- Ensures device resources are initialized exactly once
- Errors are propagated to next API call if initialization fails
- Allows graceful handling of setup failures

## Dependencies
- `core.h`: Public API declarations
- `comm.h`: Communicator structure definitions
- `nccl_device/impl/*__funcs.h`: Device-side function declarations

## Integration
This API is used by:
1. **Application Code**: Direct calls to team management functions
2. **Collective Implementations**: Create barriers and LLA2A handles for symmetric kernels
3. **Scheduler**: Allocates device resources based on requirements
4. **Transport Layer**: Initializes device state during communicator setup

The device API bridges host-side resource management with device-side execution:
- Host calculates requirements and allocates memory
- Device kernels access pre-allocated resources via handles
- No dynamic allocation in device code ensures deterministic performance
