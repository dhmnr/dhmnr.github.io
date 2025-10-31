+++
date = '2025-10-27T00:20:27-07:00'
draft = true
title = 'Speedrunning GPU Profiling with Nsight Compute CLI'
+++

{{< speedrun-def >}}

You've written your first CUDA kernel. You even implemented a basic reduction. It runs. But is it fast? Is it optimized? Where are the bottlenecks? How do you even answer these question? 

<br>

The Answer is Profiling.


Theere's a famous quote by Peter Drucker (although in a completely different context) that applies here. 
"If you can't measure it, you can't improve it". So in order to improve our kernel we must peek through the curtain to see what's actually happening. In this guide we will speedrun NVIDIA Nsight compute or `ncu`.


<br>

Before we start, as you might have guessed this post requires basic CUDA and GPU architecture knowledge. For everything else we'll try to explain the concepts as we go.

Also we will only cover the CLI part of ncu, there's a nice UI that comes bundled with it which is **not** be covered. Don't worry though, CLI is more than enough.


Let's dive in! 

<!-- ---

## Table of Contents

1. [Getting Started with Nsight Compute](#getting-started)
2. [Understanding the Output Sections](#output-sections)
3. [The Critical Metrics](#critical-metrics)
4. [Identifying Bottlenecks](#identifying-bottlenecks)
5. [Optimization Workflows](#optimization-workflows)
6. [Case Study: Vector Addition](#case-study)
7. [Common Patterns](#common-patterns)
8. [Advanced Metrics](#advanced-metrics)
9. [Troubleshooting](#troubleshooting)

--- -->

## Getting Started with Nsight Compute {#getting-started}

### Installation

Nsight Compute or `ncu` should come installed with your CUDA Toolkit. Check your installation by running `ncu --version`.

```
$ ncu --version
NVIDIA (R) Nsight Compute Command Line Profiler
Copyright (c) 2018-2025 NVIDIA Corporation
Version 2025.3.1.0 (build 36398880) (public-release)
```
If it's not installed, grab it from https://developer.nvidia.com/tools-overview/nsight-compute/get-started

### Basic Usage

Profiling CUDA kernels with `ncu` is as simple as running:

```
ncu ./your_program
```
Which profiles you program with the default metric set. In case you wanted all the details in the world, run:

```
ncu --set full ./your_program
```
You have multiple Kernels? Don't worry. Just specify the `--kernel-name`.

```
ncu --kernel-name vectorAddKernel ./your_program
```
Generate and save your report for analysis (more on this later).
```
ncu -o report ./your_program
```
<!-- 
### Common Options

```bash
# Profile first 3 kernel launches only
ncu --launch-count 3 ./program

# Target specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./program

# Export as CSV
ncu --csv ./program > metrics.csv

# Quiet output (less verbose)
ncu --quiet ./program
``` -->

---

## Understanding the Output Sections {#output-sections}

Alright, Let's do a real profiling session. Here's a simple CUDA kernel that adds two vectors together:

```
__global__ void vectorAddKernel(const float *a, const float *b, float *c,
                                int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Grid-stride loop for handling arrays larger than grid size
  for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
    c[i] = a[i] + b[i];
  }
}
```
Simple, strightforward implementation. Let's compile and profile our executable with `ncu`. We get the following output:

```
$ ncu vector_add.exe
==PROF== Connected to process 37136
==PROF== Profiling "vectorAddKernel" - 0: 0%....50%....100% - 8 passes
==PROF== Disconnected from process 37136
[37136] vector_add.exe@127.0.0.1
  vectorAddKernel(const float *, const float *, float *, int) 
  (65535, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        10.24
    SM Frequency                    Ghz         2.19
    Elapsed Cycles                cycle      398,420
    Memory Throughput                 %        95.46
    DRAM Throughput                   %        95.46
    Duration                         us       181.09
    L1/TEX Cache Throughput           %         8.88
    L2 Cache Throughput               %        33.27
    SM Active Cycles              cycle   388,019.69
    Compute (SM) Throughput           %         6.95
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute 
          or memory performance of the device.
          To further improve performance, work will likely need to be shifted from 
          the most utilized to another unit.
          Start by analyzing DRAM in the Memory Workload Analysis section.                                      


    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                 65,535
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte           16.38
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM             128
    Stack Size                                                 1,024
    Threads                                   thread      16,776,960
    # TPCs                                                        64
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                               85.33
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           24
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        89.04
    Achieved Active Warps Per SM           warp        42.74
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 10.96%                                                                            

          The difference between calculated theoretical (100.0%) and measured 
          achieved occupancy (89.0%) can be the result of warp scheduling overheads 
          or workload imbalances during the kernel execution. Load imbalances can 
          occur between warps within a block as well as across blocks of the same 
          kernel. See the CUDA Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) 
          for more details on optimizing occupancy.


    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle 1,770,174.67
    Total DRAM Elapsed Cycles        cycle   22,252,544
    Average L1 Active Cycles         cycle   388,019.69
    Total L1 Elapsed Cycles          cycle   60,391,220
    Average L2 Active Cycles         cycle   344,681.31
    Total L2 Elapsed Cycles          cycle   12,606,876
    Average SM Active Cycles         cycle   388,019.69
    Total SM Elapsed Cycles          cycle   60,391,220
    Average SMSP Active Cycles       cycle   387,318.67
    Total SMSP Elapsed Cycles        cycle  241,564,880
    -------------------------- ----------- ------------
```

Nsight Compute organizes output into **sections**. With default set we get four sections.
 - GPU Speed Of Light Throughput
 - Launch Statistics
 - Occupancy
 - GPU and Memory Workload Distribution

Let's break down each one.

---

### GPU Speed Of Light Throughput


This is the **most important section**, It tells you what's limiting your kernel.

```
Section: GPU Speed Of Light Throughput
----------------------- ----------- ------------
Metric Name             Metric Unit Metric Value
----------------------- ----------- ------------
DRAM Frequency                  Ghz        10.24
SM Frequency                    Ghz         2.23
Elapsed Cycles                cycle      474,348
Memory Throughput                 %        94.30  ← KEY!
DRAM Throughput                   %        94.30  ← KEY!
Duration                         us       212.61
L1/TEX Cache Throughput           %         8.85
L2 Cache Throughput               %        33.81
SM Active Cycles              cycle   390,062.12
Compute (SM) Throughput           %         6.92  ← KEY!
----------------------- ----------- ------------
```

### Metric Breakdown

#### 🔥 **Memory Throughput: 94.30%**

```
What it means:
  You're using 94.3% of theoretical peak memory bandwidth
  
How to interpret:
  ✅ 80-100%:  Memory is saturated (bottleneck!)
  ⚠️  50-80%:  Memory is busy but not maxed
  ✅ 0-50%:   Memory is not the bottleneck

Your value: 94.30% → MEMORY-BOUND kernel ✅
```

**Translation:** Your kernel is waiting for data from memory. It's maxed out on memory bandwidth.

#### 🔥 **Compute (SM) Throughput: 6.92%**

```
What it means:
  Only 7% of your compute units are busy
  93% of compute is idle!
  
How to interpret:
  ✅ 80-100%:  Compute-bound (good for complex math)
  ⚠️  20-80%:  Mixed workload
  ⚠️  0-20%:   Compute is idle (memory-bound)

Your value: 6.92% → Confirms memory-bound
```

**Translation:** Your ALUs (arithmetic units) are sitting idle waiting for memory. The compute is trivial compared to memory access time.

#### **The Rule of Thumb**

```
High Memory Throughput (>80%) + Low Compute (<20%):
  └─ MEMORY-BOUND kernel
  └─ Optimize: Memory access patterns

Low Memory (<50%) + High Compute (>80%):
  └─ COMPUTE-BOUND kernel  
  └─ Optimize: Algorithm, use faster math

Both Low:
  └─ LATENCY-BOUND kernel
  └─ Optimize: Increase occupancy, hide latency
```

#### **Other Important Metrics**

**Elapsed Cycles: 474,348**
```
Total cycles the kernel took
Duration = Elapsed Cycles / SM Frequency
474,348 / 2.23 GHz = 212.61 μs ✅
```

**L1/TEX Cache Throughput: 8.85%**
```
Low value → Not much data reuse from L1
Expected for streaming workloads
```

**L2 Cache Throughput: 33.81%**
```
Moderate usage of L2 cache
Some data reuse happening
```

### 📊 Interpretation Summary

```
┌─────────────────────────────────────────────────┐
│  Your Kernel Profile:                           │
├─────────────────────────────────────────────────┤
│  Memory Throughput:  94.30% → SATURATED! 🔴    │
│  Compute Throughput:  6.92% → Idle 🟢          │
│  Conclusion: MEMORY-BOUND                       │
│                                                  │
│  What this means:                               │
│  • Memory is the bottleneck                     │
│  • Can't go much faster without:               │
│    - Better memory access patterns              │
│    - Different algorithm                        │
│    - Faster GPU with more bandwidth             │
└─────────────────────────────────────────────────┘
```

---

## Section 2: Launch Statistics

### What It Shows

Configuration of your kernel launch - how you set up the grid and blocks.

### Example Output

```
Section: Launch Statistics
-------------------------------- --------------- ---------------
Metric Name                          Metric Unit    Metric Value
-------------------------------- --------------- ---------------
Block Size                                                   256  ← Threads per block
Function Cache Configuration                     CachePreferNone
Grid Size                                                 65,535  ← Number of blocks
Registers Per Thread             register/thread              16  ← Resource usage
Shared Memory Configuration Size           Kbyte           16.38
Driver Shared Memory Per Block       Kbyte/block            1.02
Dynamic Shared Memory Per Block       byte/block               0  ← You're not using any
Static Shared Memory Per Block        byte/block               0
# SMs                                         SM             128  ← GPU has 128 SMs
Stack Size                                                 1,024
Threads                                   thread      16,776,960  ← Total threads
# TPCs                                                        64
Enabled TPC IDs                                              all
Uses Green Context                                             0
Waves Per SM                                               85.33  ← Important!
-------------------------------- --------------- ---------------
```

### Key Metrics Explained

#### **Block Size: 256**

```
Your choice when launching kernel:
kernel<<<grid, block>>>(...);
           └─ block = 256 threads

Guidelines:
  ✅ Multiple of 32 (warp size)
  ✅ 128-512 is typical
  ⚠️  Too small (<64): Underutilizes GPU
  ⚠️  Too large (>1024): May limit occupancy

Your value: 256 ✅ Good choice!
```

#### **Grid Size: 65,535**

```
Number of blocks launched:
kernel<<<grid, block>>>(...);
           └─ grid = 65,535 blocks

Total threads = Grid × Block
              = 65,535 × 256
              = 16,776,960 threads ✅
```

#### **Registers Per Thread: 16**

```
How many registers each thread uses

Impact on occupancy:
  ✅ <32:  Excellent (low resource usage)
  ✅ 32-64: Good
  ⚠️  64-128: May limit occupancy
  ❌ >128: Likely limits occupancy

Your value: 16 ✅ Excellent! Very low resource usage
```

**Why it matters:** Registers are a limited resource. The RTX 4090 has 65,536 registers per SM. If each thread uses many registers, fewer threads can fit on the SM.

#### **Shared Memory Per Block: 0 bytes**

```
Dynamic + Static = 0 + 0 = 0 bytes used

You're not using shared memory (yet!)
Opportunity for optimization if data is reused
```

#### 🔥 **Waves Per SM: 85.33**

```
What it means:
  Each SM processes 85.33 "waves" of blocks
  
Calculation:
  Total blocks: 65,535
  SMs: 128
  Blocks per SM: 65,535 / 128 = 511.8
  
  Max concurrent blocks per SM: 6 (from occupancy limits)
  Waves: 511.8 / 6 = 85.3 ✅

Interpretation:
  High wave count → Work well distributed
  Low wave count → May have load imbalance
```

### Configuration Sanity Checks

```
✅ Block size is multiple of 32
✅ Total threads ≥ data elements (16M elements)
✅ Registers per thread is low (<32)
✅ Waves > 1 (workload is significant)
✅ Grid size doesn't exceed limits (max 2^31-1)
```

---

## Section 3: Occupancy

### What It Shows

**How well you're utilizing the GPU's ability to run multiple threads/warps simultaneously.**

This is **critical** for hiding memory latency!

### Example Output

```
Section: Occupancy
------------------------------- ----------- ------------
Metric Name                     Metric Unit Metric Value
------------------------------- ----------- ------------
Block Limit SM                        block           24
Block Limit Registers                 block           16  ← Limited by registers
Block Limit Shared Mem                block           16
Block Limit Warps                     block            6  ← LIMITING FACTOR!
Theoretical Active Warps per SM        warp           48
Theoretical Occupancy                     %          100
Achieved Occupancy                        %        87.89  ← Actual performance
Achieved Active Warps Per SM           warp        42.19
------------------------------- ----------- ------------

OPT   Est. Local Speedup: 12.11%
      The difference between theoretical (100.0%) and achieved (87.9%) 
      can be the result of warp scheduling overheads or workload 
      imbalances during kernel execution.
```

### Understanding Occupancy Limits

#### **Block Limits** (What could fit?)

Your GPU could theoretically fit:
```
Block Limit SM:          24 blocks (if no other constraints)
Block Limit Registers:   16 blocks (limited by register usage)
Block Limit Shared Mem:  16 blocks (limited by shared memory)
Block Limit Warps:        6 blocks ← THE LIMITING FACTOR!
```

**The bottleneck:** Your block size of 256 threads = 8 warps per block. The SM can hold max 48 warps, so: `48 warps / 8 warps per block = 6 blocks max` **← This is the constraint!**

#### **Theoretical Occupancy: 100%**

```
If everything was perfect:
  Max warps per SM: 48
  Your block could fill: 6 blocks × 8 warps = 48 warps
  Theoretical: 48 / 48 = 100%
```

#### 🔥 **Achieved Occupancy: 87.89%**

```
What actually happened:
  Average active warps: 42.19 (out of 48 max)
  Achieved: 42.19 / 48 = 87.89%

Rating:
  ✅ >90%:  Excellent
  ✅ 75-90%: Very good (your case!)
  ⚠️  50-75%: Good, room for improvement
  ⚠️  25-50%: Moderate, likely performance issues
  ❌ <25%:  Poor, definitely a problem

Your value: 87.89% → VERY GOOD! ✅
```

### Why Not 100%?

The **12% gap** comes from:

```
1. Memory Stalls (biggest factor)
   └─ Warps waiting for data from DRAM
   └─ ~200-300 cycle latency per memory access
   
2. Warp Scheduling Overhead
   └─ GPU scheduler needs cycles to switch warps
   
3. Synchronization Points
   └─ __syncthreads() causes warps to wait
   
4. Instruction Dependencies
   └─ Some instructions depend on previous results
   
5. Workload Imbalance
   └─ Some warps finish early, others keep running
```

### The 12% Potential Speedup

```
OPT   Est. Local Speedup: 12.11%
```

**What this means:** If you could achieve 100% occupancy (perfect), you might see up to 12% speedup. However, this is theoretical and often not achievable.

### Optimization Strategies for Occupancy

```
To increase occupancy:

1. Reduce register usage
   └─ Use -maxrregcount=N compiler flag
   └─ Simplify computation
   └─ Split into multiple kernels

2. Reduce shared memory usage
   └─ Use smaller tiles
   └─ Recalculate instead of store

3. Adjust block size
   └─ Try 128, 256, 512 threads
   └─ Profile each to find sweet spot

4. Increase workload size
   └─ More blocks → better GPU utilization
```

**Important:** Don't chase 100% occupancy blindly! 
- 87% is already very good
- Optimizing further may hurt other aspects
- Balance occupancy with memory coalescing

---

## Section 4: GPU and Memory Workload Distribution

### What It Shows

How different parts of the GPU (DRAM, L1, L2, SMs) were utilized during execution.

### Example Output

```
Section: GPU and Memory Workload Distribution
--------------------------- ----------- ------------
Metric Name                 Metric Unit Metric Value
--------------------------- ----------- ------------
Average DRAM Active Cycles        cycle 2,052,717.33  ← DRAM was busy
Total DRAM Elapsed Cycles         cycle  26,121,216   ← Total time available
Average L1 Active Cycles          cycle   390,062.12
Total L1 Elapsed Cycles           cycle  60,640,428
Average L2 Active Cycles          cycle   344,730.67
Total L2 Elapsed Cycles           cycle  14,909,940
Average SM Active Cycles          cycle   390,062.12
Total SM Elapsed Cycles           cycle  60,640,428
Average SMSP Active Cycles        cycle   389,666.42
Total SMSP Elapsed Cycles         cycle 242,561,712
--------------------------- ----------- ------------
```

### Understanding Active vs Elapsed

#### **Concept:**

```
Elapsed Cycles: Total time the unit was "available"
Active Cycles:  Time the unit was actually "doing work"

Utilization = Active / (Elapsed / # Units)
```

#### **DRAM Utilization**

```
Average DRAM Active Cycles:  2,052,717.33
Total DRAM Elapsed Cycles:   26,121,216

RTX 4090 has 12 memory partitions (384-bit / 32-bit)
Per partition: 26,121,216 / 12 = 2,176,768 cycles

Utilization: 2,052,717 / 2,176,768 = 94.3%

This matches "Memory Throughput: 94.30%" ✅
```

**Interpretation:** Your DRAM is **94% saturated**. Memory is definitely the bottleneck!

#### **L1 Cache Utilization**

```
Active: 390,062 cycles
Elapsed: 60,640,428 cycles (across all L1 caches)

Low utilization → Not much L1 cache reuse
Expected for streaming memory patterns
```

#### **SM (Compute) Utilization**

```
Active: 390,062 cycles (doing compute)
Elapsed: 60,640,428 cycles (available time)

Low utilization → Compute is idle most of the time
Confirms compute-bound observation (6.92%)
```

### What These Numbers Tell You

```
┌──────────────────────────────────────────────┐
│  Workload Analysis                           │
├──────────────────────────────────────────────┤
│  DRAM:    94% utilized → BOTTLENECK! 🔴     │
│  L1:      Low activity → No L1 reuse 🟡     │
│  L2:      33% utilized → Some L2 reuse 🟢   │
│  SM:      Low activity → Waiting for data 🟡│
│                                               │
│  Conclusion: Memory-bound, streaming workload│
└──────────────────────────────────────────────┘
```

---

## The Critical Metrics (Quick Reference) {#critical-metrics}

### The Big Three 🎯

```
1. Memory Throughput
   └─ >80% → Memory-bound ✅ for simple ops
   └─ <50% → Not memory-bound

2. Compute Throughput  
   └─ >80% → Compute-bound ✅ for complex math
   └─ <20% → Compute is idle

3. Achieved Occupancy
   └─ >75% → Good ✅
   └─ <50% → May have issues
```

### Decision Tree

```
                 Start Here
                     │
          ┌──────────┴──────────┐
          │                     │
    Memory Thru >80%?     Compute Thru >80%?
          │                     │
         YES                   YES
          │                     │
    MEMORY-BOUND          COMPUTE-BOUND
          │                     │
    Optimize:              Optimize:
    • Access patterns      • Algorithm
    • Coalescing          • Use faster math
    • Cache reuse         • Tensor cores
          │                     │
          └──────────┬──────────┘
                     │
              Both <80%?
                     │
                    YES
                     │
              LATENCY-BOUND
                     │
              Optimize:
              • Increase occupancy
              • More blocks/threads
              • Hide latency
```

---

## Identifying Bottlenecks {#identifying-bottlenecks}

### Memory-Bound Pattern

```
Symptoms:
  ✓ Memory Throughput: >80%
  ✓ Compute Throughput: <20%
  ✓ Low L1/L2 cache hit rates
  ✓ Simple arithmetic operations

Example: Vector addition, memory copies, reductions

Solutions:
  1. Ensure coalesced memory access
  2. Use shared memory for reuse
  3. Increase arithmetic intensity
  4. Fuse multiple operations
```

### Compute-Bound Pattern

```
Symptoms:
  ✓ Compute Throughput: >80%
  ✓ Memory Throughput: <50%
  ✓ Complex math operations
  ✓ Many FLOPs per memory access

Example: Matrix multiplication, FFT, complex math

Solutions:
  1. Use Tensor Cores (for matrix ops)
  2. Use faster math (-use_fast_math)
  3. Optimize algorithm
  4. Use specialized libraries (cuBLAS, cuFFT)
```

### Latency-Bound Pattern

```
Symptoms:
  ✓ Both Memory & Compute: <50%
  ✓ Low occupancy (<50%)
  ✓ Small workload size
  ✓ Too few blocks/threads

Example: Small data, few threads, frequent sync

Solutions:
  1. Increase occupancy
     └─ More blocks, larger grid
  2. Reduce resource usage
     └─ Fewer registers, less shared memory
  3. Reduce synchronization
  4. Batch multiple operations
```

### Occupancy-Limited Pattern

```
Symptoms:
  ✓ Achieved Occupancy: <50%
  ✓ High register usage (>64 per thread)
  ✓ High shared memory usage (>48 KB per block)
  ✓ Large block size (>512 threads)

Solutions:
  1. Reduce registers
     └─ -maxrregcount=N flag
  2. Reduce shared memory
     └─ Smaller tiles
  3. Adjust block size
     └─ Try different sizes (128, 256, 512)
  4. Launch more blocks
```

---

## Case Study: Vector Addition {#case-study}

Let's analyze a complete profiling session step-by-step.

### The Kernel

```cuda
__global__ void vectorAddKernel(const float *a, const float *b, 
                                float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Launch: 16M elements
int n = 1 << 24;  // 16,777,216 elements
int blockSize = 256;
int gridSize = (n + blockSize - 1) / blockSize;  // 65,536 blocks
vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
```

### The Profiling Command

```bash
ncu --set full ./vector_add
```

### Analysis Walkthrough

#### Step 1: Check Speed of Light

```
Memory Throughput:  94.30% 🔴
Compute Throughput:  6.92% 🟢

Immediate conclusion: MEMORY-BOUND!
```

**Why this makes sense:**
```
Operation per element:
  • Read a[i]:  4 bytes, ~200 cycles
  • Read b[i]:  4 bytes, ~200 cycles  
  • Add:        1 cycle (trivial!)
  • Write c[i]: 4 bytes, ~200 cycles

Memory dominates: 3 × 200 = 600 cycles
Compute is tiny: 1 cycle

Ratio: 600:1 memory to compute!
```

#### Step 2: Check Launch Configuration

```
Block Size:           256 ✅ Good (multiple of 32)
Grid Size:            65,535 ✅ Enough blocks
Registers Per Thread: 16 ✅ Excellent (very low)
Shared Memory:        0 bytes ✅ Not needed here
Waves Per SM:         85.33 ✅ High (well distributed)
```

**Assessment:** Launch configuration is optimal!

#### Step 3: Check Occupancy

```
Theoretical Occupancy: 100%
Achieved Occupancy:    87.89% ✅

Gap: 12.11%
Reason: Memory stalls (warps waiting for data)
```

**Assessment:** Occupancy is very good. The 12% gap is expected for memory-bound kernels.

#### Step 4: Check Memory Workload

```
DRAM Active/Elapsed: 94.3% 🔴 Saturated!
L1 Activity:         Low 🟡 (streaming, no reuse)
L2 Activity:         33% 🟢 (some reuse)
SM Activity:         Low 🟡 (waiting for memory)
```

**Assessment:** DRAM is the clear bottleneck.

#### Step 5: Check Memory Access Pattern

```
(Would need more detailed metrics, but we know from code:)

Memory Access Pattern:
  idx = blockIdx.x * blockDim.x + threadIdx.x;
  c[idx] = a[idx] + b[idx];
  
Sequential access! ✅
  └─ Perfect coalescing
  └─ All warps access consecutive addresses
```

### Final Verdict

```
┌─────────────────────────────────────────────────┐
│  Vector Addition Performance Summary            │
├─────────────────────────────────────────────────┤
│  Status:      OPTIMAL ✅                        │
│  Bottleneck:  Memory bandwidth (expected)       │
│  Achieved:    94% of theoretical peak           │
│                                                  │
│  Why optimal:                                   │
│  ✓ Sequential access (coalesced)               │
│  ✓ Low register usage                           │
│  ✓ Good occupancy (87%)                         │
│  ✓ Memory saturated (can't go faster)          │
│                                                  │
│  Can we optimize further?                       │
│  └─ Not without changing algorithm              │
│      Options:                                   │
│      • Fuse with other operations               │
│      • Use faster GPU with more bandwidth       │
│      • Accept that vector add is memory-bound   │
└─────────────────────────────────────────────────┘
```

### CPU vs GPU Timing Mystery Solved

**User's original question:**
```
Time (CPU timing):      92.2359 ms
Bandwidth calculated:   2.03 GB/s  ← Seems terrible?

Time (GPU kernel only): 0.21261 ms
Memory throughput:      94.3% of 1,008 GB/s = 950 GB/s ← Actually excellent!
```

**What happened?**

```
CPU timing includes EVERYTHING:
  ├─ cudaMalloc            ~1 ms
  ├─ Host→Device transfer  ~40 ms (64 MB over PCIe)
  ├─ Kernel execution      ~0.21 ms ← Only this is GPU work!
  ├─ Device→Host transfer  ~40 ms
  ├─ cudaDeviceSynchronize ~10 ms
  └─ Other overhead        ~1 ms
  ──────────────────────
  Total: ~92 ms

The kernel itself is incredibly fast (0.21 ms)!
But data transfers dominate the total time.
```

**Lesson:** Always profile the kernel specifically, not just end-to-end timing!

---

## Common Patterns and What They Mean {#common-patterns}

### Pattern 1: Perfect Compute-Bound

```
Memory Throughput:  15%
Compute Throughput: 95%
Achieved Occupancy: 85%

Interpretation:
  • Kernel is doing heavy computation
  • Memory is not the bottleneck
  • Likely complex math operations
  
Example: Matrix multiplication, convolution, FFT

Optimization:
  ✓ This is already good for compute-heavy workloads
  • Consider using Tensor Cores
  • Use -use_fast_math for approximations
  • Check if specialized libraries exist
```

### Pattern 2: Terrible Access Pattern

```
Memory Throughput:  30%
Compute Throughput:  5%
Achieved Occupancy: 90%
L1/L2 Hit Rate:     <10%

Interpretation:
  • Memory is being accessed inefficiently
  • Likely random or strided access
  • Poor coalescing
  • Cache thrashing
  
Example: Random lookups, large strides, transpose

Optimization:
  ❌ Fix memory access pattern immediately!
  • Use sequential access
  • Use shared memory for complex patterns
  • Consider data layout changes (AoS → SoA)
```

### Pattern 3: Occupancy Problem

```
Memory Throughput:  40%
Compute Throughput: 30%
Achieved Occupancy: 25%  ← TOO LOW!
Registers Per Thread: 128

Interpretation:
  • Too few warps active
  • High register usage limiting occupancy
  • GPU is underutilized
  
Optimization:
  1. Reduce register usage
     └─ nvcc -maxrregcount=64
  2. Increase block/grid size
  3. Split complex kernel into multiple passes
```

### Pattern 4: Small Workload

```
Memory Throughput:  25%
Compute Throughput: 20%
Achieved Occupancy: 90%
Waves Per SM:       0.5  ← Very low!

Interpretation:
  • Workload is too small
  • GPU is underutilized
  • Kernel launch overhead dominates
  
Optimization:
  • Increase problem size
  • Batch multiple operations
  • Consider CPU for small workloads
  • Fuse multiple kernels
```

### Pattern 5: Perfect Balance

```
Memory Throughput:  65%
Compute Throughput: 75%
Achieved Occupancy: 85%
L2 Hit Rate:        50%

Interpretation:
  • Balanced workload
  • Good cache utilization
  • Both memory and compute are utilized
  
Example: Well-optimized matrix operations, 
         kernels with good arithmetic intensity

This is the sweet spot! ✅
```

---

## Optimization Workflows {#optimization-workflows}

### Workflow 1: Memory-Bound Kernel

```
Step 1: Confirm memory is the bottleneck
  └─ Memory Throughput >80%? ✓

Step 2: Check access pattern
  ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum

  Ideal: sectors_read ≈ (bytes_read / 32)
  If much higher: Poor coalescing!

Step 3: Check L2 cache hit rate
  ncu --metrics lts__t_sector_hit_rate.pct
  
  <30%: No data reuse
  30-70%: Some reuse
  >70%: Good reuse

Step 4: Optimize
  A. Fix coalescing (sequential access)
  B. Use shared memory for reused data
  C. Increase arithmetic intensity (fuse ops)
  D. Consider different algorithm

Step 5: Re-profile and compare
```

### Workflow 2: Occupancy-Limited

```
Step 1: Identify limitation
  Look at Block Limit metrics:
  • Block Limit Registers: X
  • Block Limit Shared Mem: Y
  • Block Limit Warps: Z
  
  Minimum is your constraint!

Step 2: If limited by registers
  A. Compile with -maxrregcount=N
  B. Simplify computation
  C. Split into multiple kernels
  D. Use more blocks with fewer threads

Step 3: If limited by shared memory
  A. Reduce tile sizes
  B. Use multiple kernel passes
  C. Recalculate instead of storing

Step 4: If limited by warps
  A. Reduce block size
  B. Increase number of blocks

Step 5: Re-profile and check
  └─ Achieved Occupancy improved?
```

### Workflow 3: Compute-Bound Kernel

```
Step 1: Confirm compute is bottleneck
  └─ Compute Throughput >80%? ✓

Step 2: Check if math can be approximated
  • sin, cos, log, exp → use fast versions
  • Compile with -use_fast_math

Step 3: Check for Tensor Core opportunity
  • Matrix multiplication?
  • Use WMMA or cuBLAS

Step 4: Check instruction mix
  ncu --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
  
  • Lots of transcendentals? Consider lookup tables
  • Integer/FP conversion overhead? Reduce conversions

Step 5: Profile algorithm complexity
  • Can you reduce operations?
  • Better algorithm exists?
```

---

## Advanced Metrics {#advanced-metrics}

### Memory Coalescing Metrics

```bash
# Check memory transaction efficiency
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
              l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
    ./program
```

**Interpretation:**
```
Ideal coalescing:
  Sectors ≈ (Total Bytes Accessed / 32)

Poor coalescing:
  Sectors >> (Total Bytes Accessed / 32)
  └─ You're loading extra data unnecessarily
```

### Cache Hit Rates

```bash
# L2 cache hit rate
ncu --metrics lts__t_sector_hit_rate.pct ./program

# L1 cache hit rate  
ncu --metrics l1tex__t_sector_hit_rate.pct ./program
```

**What good looks like:**
```
L2 Hit Rate:
  >70%: Excellent data reuse
  40-70%: Good
  <40%: Poor reuse (streaming workload)

L1 Hit Rate:
  >50%: Excellent
  20-50%: Moderate
  <20%: Poor (but OK for streaming)
```

### Warp Execution Efficiency

```bash
# Check for divergence
ncu --metrics smsp__average_warp_latency_per_inst_executed.ratio \
    ./program
```

**Interpretation:**
```
Low value: Good (no divergence)
High value: Warps are diverging
  └─ Branches causing serialization
  └─ Review if-statements
```

### Roofline Model

```bash
# Get arithmetic intensity
ncu --metrics sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
              sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
              dram__bytes.sum \
    ./program
```

**Calculate:**
```
FLOPs = fadd + fmul + fma*2
Bytes = dram__bytes.sum
Arithmetic Intensity = FLOPs / Bytes

Compare to GPU's Balance Point:
  AI < Peak_FLOPS / Peak_BW → Memory-bound
  AI > Peak_FLOPS / Peak_BW → Compute-bound
```

---

## Troubleshooting Common Issues {#troubleshooting}

### Issue 1: No Output or "Permission Denied"

```bash
# On Linux, need admin privileges
sudo ncu ./program

# Or set capabilities (one time setup)
sudo setcap cap_sys_admin=ep $(which ncu)
```

### Issue 2: "Too Many Metrics"

```bash
# Profiling might take many passes
# Reduce metric count:
ncu --set basic ./program      # Fewer metrics
ncu --set full ./program       # All metrics (slow)

# Or specific metrics only:
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./program
```

### Issue 3: Kernel Runs Multiple Times

```
This is normal! ncu runs kernels multiple times to collect metrics.

==PROF== Profiling "kernel" - 0: 0%....50%....100% - 8 passes
                                                        └─ 8 runs!

Use --launch-count to limit:
ncu --launch-count 1 ./program
```

### Issue 4: Can't Find Specific Kernel

```bash
# List all kernels first
ncu --list-kernels ./program

# Then target specific one
ncu --kernel-name "vectorAddKernel" ./program

# Or by index
ncu --launch-count 1 --kernel-id 0 ./program
```

### Issue 5: Output Too Verbose

```bash
# Quiet mode
ncu --quiet ./program

# Or redirect to file
ncu ./program > profile_output.txt 2>&1
```

### Issue 6: Numbers Don't Match Documentation

```
This is normal! 
• Metrics vary by GPU architecture
• Different CUDA versions report differently
• Workload affects measurements

Always compare:
  • Same GPU
  • Same CUDA version
  • Similar workload size
```

---

## Practical Tips and Best Practices

### General Profiling Strategy

```
1. Start with high-level metrics (Speed of Light)
2. Identify the bottleneck
3. Dive into specific metrics
4. Make targeted changes
5. Re-profile and compare

DON'T:
  ❌ Try to optimize everything at once
  ❌ Chase 100% on all metrics
  ❌ Optimize without profiling
  ❌ Focus on micro-optimizations first
```

### Interpreting "Good" vs "Bad"

```
Context matters!

Memory-Bound Kernel:
  ✅ 90% memory throughput is EXCELLENT
  ✅ 5% compute throughput is EXPECTED
  
Compute-Bound Kernel:
  ✅ 90% compute throughput is EXCELLENT  
  ✅ 20% memory throughput is EXPECTED

Don't aim for 100% on everything!
Each workload has a natural profile.
```

### Using the GUI

```bash
# Generate report
ncu -o my_report ./program

# Open in GUI (better visualization)
ncu-ui my_report.ncu-rep

Benefits of GUI:
  • Visual charts
  • Side-by-side comparison
  • Source code correlation
  • Easier navigation
```

### Comparing Before/After

```bash
# Before optimization
ncu -o before ./program

# After optimization
ncu -o after ./program

# Compare in GUI
ncu-ui before.ncu-rep after.ncu-rep

Look for:
  • Memory throughput improvement
  • Occupancy changes
  • Duration reduction
```

### Automated Analysis

```bash
# Export metrics as CSV
ncu --csv ./program > metrics.csv

# Then process with Python/scripts
# Good for regression testing
```

---

## Quick Reference Card

### Essential Commands

```bash
# Basic profile
ncu ./program

# Full metrics
ncu --set full ./program

# Specific metrics
ncu --metrics METRIC_NAME ./program

# Save report
ncu -o report ./program

# Target specific kernel
ncu --kernel-name NAME ./program

# First launch only
ncu --launch-count 1 ./program
```

### Key Metrics Interpretation

```
Memory Throughput:
  >80% → Memory-bound ✅ (for simple ops)
  <50% → Not memory-bound

Compute Throughput:
  >80% → Compute-bound ✅ (for complex math)
  <20% → Compute idle

Achieved Occupancy:
  >85% → Excellent ✅
  >75% → Very good ✅
  >50% → Good
  <50% → Needs improvement

Registers Per Thread:
  <32 → Excellent ✅
  <64 → Good ✅
  <128 → Okay
  >128 → Likely problematic
```

### Bottleneck Quick Check

```
IF Memory Thru >80% AND Compute Thru <20%:
  → MEMORY-BOUND
  → Optimize: Access patterns, coalescing, cache
  
ELSE IF Compute Thru >80% AND Memory Thru <50%:
  → COMPUTE-BOUND
  → Optimize: Algorithm, fast math, Tensor Cores
  
ELSE IF Both <50% AND Occupancy <50%:
  → OCCUPANCY-BOUND
  → Optimize: Resources, block size, more blocks
  
ELSE:
  → BALANCED or LATENCY-BOUND
  → Check: Synchronization, divergence, small workload
```

---

## Conclusion

NVIDIA Nsight Compute is incredibly powerful, but the output can be overwhelming. The key is knowing **which metrics matter** and **how to interpret them**.

### Remember the Fundamentals:

1. **Start with Speed of Light** - Tells you the big picture
2. **Memory vs Compute throughput** - Identifies the bottleneck
3. **Occupancy matters** - But don't chase 100%
4. **Context is everything** - "Good" depends on workload type

### The Optimization Loop:

```
Profile → Identify Bottleneck → Optimize → Re-Profile → Compare

Repeat until:
  • Performance goal achieved, OR
  • Metrics show optimal utilization for workload type
```

### When to Stop Optimizing:

You've reached the limit when:
- ✅ Bottleneck metric >90% (memory or compute)
- ✅ Occupancy >75%
- ✅ Access patterns are optimal
- ✅ Profiler shows no obvious issues

At this point, further improvement requires algorithmic changes, not just code tweaks.

---

## Further Resources

**Official Documentation:**
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA C Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Profiler User's Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)

**Interactive Learning:**
- NVIDIA's Nsight Compute training videos
- CUDA profiling webinars
- GTC talks on performance optimization

**Community:**
- NVIDIA Developer Forums
- Stack Overflow (#cuda, #nvidia-nsight)
- GPU computing Discord/Slack communities

---

**Happy Profiling!** 🚀

Now you have the knowledge to understand what your GPU is really doing. Use ncu to guide your optimization efforts, and remember: **measure, don't guess!**

---

*Written with ❤️ for the CUDA community*  
*Last updated: 2024*  
*Target: CUDA 12.0+, Compute Capability 7.0+*


