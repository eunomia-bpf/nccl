# Scheduler - Symmetric Kernel Scheduling

## Overview
This directory implements the scheduler logic for symmetric collective operations. It groups compatible collective tasks, selects appropriate symmetric kernels, and prepares them for execution on the GPU.

## Files

### symmetric_sched.cc
**Purpose**: Orchestrates the selection and scheduling of symmetric collective kernels.

**Key Functions**:
- `ncclMakeSymmetricTaskList()`: Main entry point that partitions collective tasks into symmetric and non-symmetric queues
  - Analyzes task compatibility for symmetric execution
  - Groups tasks by function, reduction operation, and datatype
  - Selects optimal kernels for each task group
  - Allocates work descriptors within kernel argument space limits

**Algorithm Flow**:
1. **Window Discovery**: For each task, find memory windows containing send/recv buffers
2. **Eligibility Check**: Verify windows have COLL_SYMMETRIC flag and symmetric kernel is available
3. **Grouping**: Organize eligible tasks by (function, op, datatype) tuple
4. **Batching**: Pack multiple tasks into single kernel launch up to argument space limit
5. **Kernel Selection**: Call `ncclSymkPickKernel()` to select best kernel variant
6. **Queue Assignment**: Move symmetric tasks to `collSymTaskQueue`, others to remain list

**Key Concepts**:
- **Cell-based Partitioning**: Work is quantized into cells (NCCL_SYM_KERNEL_CELL_SIZE) for load balancing
- **Task Fusion**: Multiple small collectives can be fused into one kernel launch for efficiency
- **Argument Space Management**: Kernel args limited by `comm->workArgsBytes`, determines fusion limit
- **Kernel Selection**: Picks kernel ID, channel count, and warp count based on workload characteristics

## Key Concepts

### Symmetric Kernel Eligibility
Tasks qualify for symmetric kernels when:
- Send and receive buffers reside in memory windows marked COLL_SYMMETRIC
- Function/operation/datatype combination has available symmetric kernel implementation
- `ncclSymkAvailable()` returns true for the configuration

### Task Batching Strategy
Multiple tasks are batched together if:
- They share the same (function, reduction op, datatype) tuple
- Combined argument size fits within `comm->workArgsBytes` limit
- Benefits from amortizing kernel launch overhead

### Kernel Selection Criteria
`ncclSymkPickKernel()` considers:
- Total element count across all fused tasks
- Maximum element count in any single task
- Number of tasks being fused
- Returns estimated execution time, kernel ID, channel count, warp count

### Work Descriptors
Each task generates `ncclSymkDevWork` descriptor containing:
- Input/output buffer pointers and window IDs
- Element counts and data type
- Reduction operator arguments
- Channel assignment information

## Dependencies
- `scheduler.h`: Core scheduler definitions and task structures
- `ncclDevrFindWindow()`: Locates memory window containing buffer
- `ncclSymkAvailable()`: Checks if symmetric kernel exists for operation
- `ncclSymkPickKernel()`: Selects optimal kernel variant
- `ncclSymkDevWorkArgs`: Device-side work argument structure

## Integration
The symmetric scheduler is invoked as part of the overall scheduling pipeline:
1. **Task Submission**: Collective operations add tasks to planner queue
2. **Symmetric Filtering**: This module separates symmetric-eligible tasks
3. **Kernel Selection**: Assigns kernel variants to task groups
4. **Execution**: Symmetric kernels launched by kernel scheduler
5. **Fallback**: Non-symmetric tasks handled by traditional path

By leveraging symmetric memory access, these kernels achieve significantly lower latency than traditional proxy-based approaches, especially for small message sizes where launch overhead dominates.
