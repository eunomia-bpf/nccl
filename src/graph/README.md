# Topology Graph System

## Overview
The graph directory implements NCCL's topology discovery and analysis system. It builds a detailed model of the system's interconnect topology, computes optimal communication paths, and generates efficient communication patterns for collective operations.

## Files

### topo.h / topo.cc
**Purpose**: Core topology graph data structures and management.

**Node Types** (NCCL_TOPO_NODE_TYPES):
- `GPU`: NVIDIA GPU devices
- `PCI`: PCIe switches and bridges
- `NVS`: NVLink switches (NVSwitch)
- `CPU`: NUMA domains / CPU sockets
- `NIC`: Network interface cards
- `NET`: Network endpoints (remote nodes)

**Link Types**:
- `LINK_LOC`: Local (same device) - 5000 GB/s
- `LINK_NVL`: NVLink - 18-40+ GB/s depending on generation
- `LINK_C2C`: Chip-to-chip coherent link
- `LINK_PCI`: PCIe link - 12-48 GB/s depending on generation
- `LINK_SYS`: System interconnect (QPI/UPI) - 6-40 GB/s
- `LINK_NET`: Network connection - 12+ GB/s

**Path Types** (semantic distance):
- `PATH_LOC`: Same device (0)
- `PATH_NVL`: Direct NVLink (1)
- `PATH_NVB`: NVLink via intermediate GPU (2)
- `PATH_C2C`: Chip-to-chip link (3)
- `PATH_PIX`: Single PCIe bridge (4)
- `PATH_PXB`: Multiple PCIe bridges (5)
- `PATH_P2C`: CPU-to-NIC via C2C (6)
- `PATH_PXN`: Via intermediate GPU to network (7)
- `PATH_PHB`: Across PCIe host bridge/CPU (8)
- `PATH_SYS`: Across NUMA interconnect (9)
- `PATH_NET`: Across network (10)
- `PATH_DIS`: Disconnected (11)

**Key Functions**:
- Topology graph construction from system detection
- Bandwidth and latency modeling
- Path computation between any two nodes
- Affinity and locality analysis

### xml.h / xml.cc
**Purpose**: XML-based topology description parser and generator.

Enables:
- Explicit topology specification via XML files
- Override automatic detection
- Reproducible topology for testing
- Topology sharing across processes

XML format describes:
- Node hierarchy (GPUs, switches, CPUs, NICs)
- Link types and bandwidth
- Affinity relationships
- Custom topology configurations

### search.cc
**Purpose**: Graph search algorithms for optimal path finding.

Implements:
- Shortest path algorithms (bandwidth-weighted)
- Multi-path search for striping
- Rail-aware routing
- Constraint-based search (locality, bandwidth, latency)

Optimization goals:
- Maximize aggregate bandwidth
- Minimize cross-NUMA traffic
- Balance load across links
- Avoid bottlenecks

### rings.h / rings.cc
**Purpose**: Ring-based communication pattern generation.

Ring algorithm for collectives:
- Divides communicator into logical rings
- Each rank sends to next, receives from previous
- Pipelined for high bandwidth
- Good for large messages

Ring construction considers:
- GPU placement across nodes
- NVLink topology within nodes
- Network topology between nodes
- Symmetry and load balancing

### trees.cc
**Purpose**: Tree-based communication pattern generation.

Tree algorithms for collectives:
- Binary or multi-ary trees
- Reduce overhead for small messages
- Lower latency than rings
- Good for latency-sensitive operations

Tree construction considers:
- Hierarchical structure (intra-node, inter-node)
- Bandwidth at each level
- Number of rails available
- CPU/NIC affinity

### paths.cc
**Purpose**: Pre-computed path database and lookup.

Maintains:
- All-pairs shortest paths cache
- Path characteristics (type, bandwidth, hops)
- Alternative paths for multi-rail
- Locality information

### connect.cc
**Purpose**: Connection establishment based on topology.

Coordinates:
- Transport selection per connection
- Connection setup ordering
- Resource allocation
- Error handling and retry

### tuning.cc
**Purpose**: Algorithm and parameter tuning based on topology.

Determines:
- Which algorithm variant to use (ring, tree, etc.)
- Protocol selection (LL, Simple, LL128)
- Channel count and work distribution
- Chunk size and pipelining depth

Tuning considers:
- Message size
- Number of ranks
- Topology characteristics
- Hardware capabilities

## Key Concepts

### Topology Discovery
Multi-source information gathering:
1. **CUDA**: Device properties, peer access capabilities
2. **NVML**: PCIe topology, NVLink connectivity
3. **System**: NUMA configuration, CPU topology
4. **Network Plugin**: NIC locations and capabilities
5. **XML Override**: User-specified topology

### Bandwidth Modeling
Realistic bandwidth prediction:
- Link bandwidth (per-link capacity)
- Aggregate bandwidth (parallel links)
- Contention modeling (shared resources)
- Protocol overhead accounting

Example: Two GPUs on same NVSwitch:
- Raw NVLink: 25 GB/s per link
- 12 links: 300 GB/s aggregate
- Achievable: ~270 GB/s after protocol overhead

### Path Affinity
Locality awareness for optimization:
- **Intra-socket**: GPUs on same CPU socket
- **Intra-node**: GPUs on same physical machine
- **Intra-switch**: GPUs behind same network switch
- **Cross-node**: GPUs on different machines

Algorithms prefer paths with tighter affinity when possible.

### Multi-Rail Support
Multiple parallel paths between ranks:
- Each rail represented as separate path
- Load balanced across rails
- Independent progress per rail
- Aggregated effective bandwidth

Example: 8 NICs per node = 8 rails for inter-node communication

### Algorithm Selection
Topology influences algorithm choice:
- **Dense NVLink**: Prefer symmetric kernels, direct P2P
- **Sparse NVLink**: Use tree algorithms
- **Multi-node**: Prefer ring for bandwidth, tree for latency
- **Small message**: Protocol LL (low latency)
- **Large message**: Protocol Simple (high bandwidth)

### Communication Pattern Generation
Process:
1. Build topology graph from discovered information
2. Compute all-pairs paths with bandwidth/latency
3. Select algorithm (ring, tree, etc.) based on operation and size
4. Generate communication pattern matching algorithm
5. Assign channels to balance load
6. Map pattern to physical topology

Example ring generation:
1. Start with arbitrary rank
2. Select next rank maximizing intra-node links first
3. Use high-bandwidth paths where available
4. Close ring ensuring symmetry

## Dependencies
- CUDA: Device properties and topology
- NVML: PCIe and NVLink discovery
- Network plugin: NIC location and capabilities
- System calls: NUMA topology (/sys/devices/system/node)

## Integration

### Initialization
1. `ncclTopoDiscover()`: Build initial graph from system
2. `ncclTopoCompute()`: Calculate paths and characteristics
3. `ncclTopoGetAlgoTime()`: Estimate algorithm performance
4. `ncclTopoTune()`: Select optimal parameters

### Runtime
- Algorithm selection queries topology for bandwidth estimates
- Channel assignment uses topology for load balancing
- Transport selection consults topology for path types
- Tuner adjusts parameters based on topology characteristics

### Tuning Process
For each (operation, size, nRanks):
1. Query topology for path bandwidths
2. Model each algorithm's performance
3. Account for pipelining and protocol overhead
4. Select algorithm minimizing completion time
5. Cache decision for reuse

## Advanced Features

### Topology-Aware Tuning
Adaptive algorithm selection:
- DGX-1 (NVLink-only): Prefer tree algorithms
- DGX-2 (NVSwitch): Use direct patterns
- Multi-node: Hierarchical algorithms (tree intra-node, ring inter-node)
- Non-uniform: Custom patterns matching asymmetry

### Dynamic Topology
Handle topology changes:
- GPU hotplug (add/remove)
- NIC failure (switch rails)
- Link degradation (reduce bandwidth estimate)
- Recompute patterns on topology change

### XML Topology Override
Use cases:
- Testing: Simulate different topologies without hardware
- Optimization: Hand-tune for specific system
- Reproducibility: Ensure consistent behavior across runs
- Debugging: Isolate topology-related issues

### Graph Visualization
Can export topology as:
- DOT format for Graphviz
- XML for analysis tools
- Human-readable text for debugging

## Performance Impact

Topology-aware optimization improvements:
- 2-3x faster small message latency (tree vs. ring)
- 30-50% higher large message bandwidth (optimal ring construction)
- Eliminates cross-NUMA traffic in many cases
- Exploits full multi-rail bandwidth

The graph system is NCCL's "brain," using detailed topology knowledge to make intelligent decisions about communication patterns, algorithm selection, and resource allocation - essential for achieving near-optimal performance across diverse GPU cluster configurations.
