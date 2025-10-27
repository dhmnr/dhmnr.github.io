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


Let's dive in! 🚀

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

```C++
__global__ void vectorAddKernel(const float *a, const float *b, float *c,
                                int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Grid-stride loop for handling arrays larger than grid size
  for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
    c[i] = a[i] + b[i];
  }
}
```
