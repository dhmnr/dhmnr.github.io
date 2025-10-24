+++
date = '2025-10-17T23:46:20-07:00'
draft = true
title = 'Serving LLMs at Scale: FlashAttention'
+++

Attention is the bottleneck in transformer inference, but not for the reasons you might think. 
A100 sits at 30% utilization during attention not because the math is hard, but because 
it's spending most of its time waiting for data to move between memory and compute cores. 
FlashAttention fixes this by keeping computation in fast on-chip SRAM and never materializing 
the O(N²) attention matrix. Here's how it works.

<!--more-->

## The Memory Bottleneck

Standard attention's memory access pattern
Counting memory operations vs FLOPs
Why attention is memory-bound


## Understanding the Memory Hierarchy

SRAM vs HBM: capacity and bandwidth
The actual numbers that matter
Why compute sits idle


## FlashAttention's Core Insight

The tiling strategy
Online softmax algorithm
Trading recomputation for memory bandwidth


## Performance Characteristics

Speedup vs sequence length
Memory usage comparison
When it matters (and when it doesn't)


## What's Next

What FlashAttention doesn't solve
Teaser for PagedAttention

# Appendix 

## References 