# RAS (Reliability, Availability, Serviceability) System

## Overview
The RAS system provides fault detection, diagnosis, and recovery capabilities for NCCL. It monitors communicator health, detects failures, coordinates recovery actions, and provides observability into distributed training jobs.

## Files

### ras_internal.h
**Purpose**: Internal definitions for RAS protocol and data structures.

**Key Definitions**:
- `NCCL_RAS_CLIENT_PORT`: 28028 (default port for external monitoring tools)
- `RAS_COLLECTIVE_LEG_TIMEOUT_SEC`: Timeout for individual collective operations
- Message types: CONNINIT, KEEPALIVE, PEERSUPDATE, COLLREQ, COLLRESP
- Collective types: DEADPEER (broadcast), CONNS/COMMS (gather)

**Key Structures**:
- `rasCommId`: Unique communicator identifier (commHash + hostHash + pidHash)
- `rasPeerInfo`: Describes peer NCCL process (address, PID, device masks, hashes)
- `rasCollRequest`: Payload for collective requests (type, timeout, parameters)
- `rasCollResponse`: Response containing peer data and collective results

### ras.cc
**Purpose**: Core RAS infrastructure and coordination logic.

Likely implements:
- RAS thread management
- Event loop for monitoring
- Coordination of RAS collectives
- State machine for failure handling

### client.cc / client_support.cc
**Purpose**: External client interface for monitoring tools.

Enables external tools to:
- Connect to NCCL RAS port
- Query communicator status
- Retrieve diagnostic information
- Monitor job health

### rasnet.cc
**Purpose**: Network communication layer for RAS messages.

Handles:
- Peer-to-peer RAS messaging
- Connection establishment and maintenance
- Message serialization/deserialization
- Timeout handling

### collectives.cc
**Purpose**: RAS collective operations implementation.

Implements distributed operations:
- **Broadcast**: Propagate failure notifications (DEADPEER)
- **Gather**: Collect status from all ranks (CONNS, COMMS)
- Timeout-based reliability
- Partial result handling

### peers.cc
**Purpose**: Peer tracking and management.

Manages:
- Peer discovery and registration
- Liveness monitoring via keepalives
- Peer state transitions (alive, suspected, dead)
- Peer information synchronization

## Key Concepts

### RAS Thread Architecture
Each NCCL process runs a RAS thread that:
- Monitors local communicators
- Exchanges keepalive messages with peers
- Detects and reports failures
- Participates in collective recovery actions

### Failure Detection
Multiple mechanisms detect failures:
1. **Keepalive Timeout**: Peer doesn't respond within timeout window
2. **Collective Timeout**: Operation doesn't complete within expected time
3. **Transport Errors**: Network or IPC errors during communication
4. **Application Report**: User explicitly reports failure

### Collective Coordination
RAS implements reliable collectives:
- **Two-phase Protocol**: Request broadcast + response gather
- **Root Coordination**: One rank initiates, others respond
- **Timeout Handling**: Incomplete results tracked with `nLegTimeouts` counter
- **Idempotency**: Requests identified by (rootAddr, rootId) to avoid duplicates

### Communicator Identification
`rasCommId` uniquely identifies communicators across processes:
- `commHash`: Hash of communicator configuration
- `hostHash`, `pidHash`: From rank 0 to distinguish identical configs
- Sorted for efficient collective operations

### Peer State Machine
Peers transition through states:
- **Unknown**: Not yet discovered
- **Alive**: Responding to keepalives
- **Suspected**: Missed keepalive(s), not yet declared dead
- **Dead**: Confirmed failure, recovery needed

### External Client Protocol
Monitoring tools connect via TCP:
1. Send CONNINIT with protocol version
2. Receive CONNINITACK
3. Send COLLREQ for specific query
4. Receive COLLRESP with results
5. Connection closed or reused

### Data Aggregation
COMMS collective gathers:
- Communicator list per rank
- Missing rank information
- Configuration details
- Optimization: Skip redundant data with `skipMissingRanksComms` array

CONNS collective gathers:
- Active connections per rank
- Transport types in use
- Bandwidth statistics
- Error counters

## Dependencies
- `socket.h`: Network communication primitives
- `utils.h`: Host identification and hashing
- `nccl.h`: NCCL API definitions
- `comm.h`: Communicator structure

## Integration

### Initialization Flow
1. `ncclCommInitRank()` spawns RAS thread
2. RAS thread connects to peers
3. Peer discovery via bootstrap network
4. Establish keepalive protocol

### Failure Handling Flow
1. Detect failure (timeout, error, report)
2. Broadcast DEADPEER to all ranks
3. All ranks mark peer as dead
4. Trigger recovery actions in NCCL core
5. External monitoring tools notified

### Recovery Actions
When failure detected:
- Abort in-flight operations
- Mark communicator as aborted
- Propagate error to user application
- Optionally attempt recovery (future)

### Observability
External tools query:
- Live communicator list
- Rank participation status
- Error history
- Performance counters

## Design Rationale

### Separate RAS Thread
Advantages:
- Doesn't block main communication path
- Can detect failures even when main thread hung
- Provides out-of-band control plane
- Enables proactive monitoring

### Lightweight Protocol
Design optimizes for:
- Minimal overhead during normal operation
- Fast failure detection (few second timeout)
- Reliable coordination despite failures
- Support for external monitoring without code changes

### Collective-based Coordination
Benefits:
- Consistent global view of failures
- Atomic state transitions across ranks
- Simplified reasoning about distributed state
- Reuses collective algorithm infrastructure

The RAS system transforms NCCL from a "fail-stop" library into one with comprehensive monitoring, diagnosis, and potential for automatic recovery - essential for large-scale, long-running training jobs.
