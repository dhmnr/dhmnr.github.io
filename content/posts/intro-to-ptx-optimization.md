+++
date = '2025-11-12T01:09:00-08:00'
draft = false
title = 'Intro to PTX Optimization : Part 1'
+++
If you're trying to push your GPU harder than CUDA C++ allows, eventually you end up at PTX. It's NVIDIA's intermediate representation, sitting between your kernel code and the hardware.
This post covers PTX from the ground up: what it is, how to write it by hand, and why it matters. Make sure you're comfortable with CUDA kernels and the memory model before jumping in.

---

## What is PTX?

PTX is a low-level, virtual instruction set designed by NVIDIA for their GPUs. It's basically the GPU's "assembly language," but it's not the final machine code. Instead, PTX gets compiled just-in-time (JIT) by the NVIDIA driver into SASS, the actual binary that runs on your specific GPU architecture.

Why the extra step? Why not compile directly to SASS?
 - PTX written today can run on future GPUs without recompiling. Ship it once and let the driver handle new architectures as they come out.
 - Different GPUs have different instruction sets, register counts, and quirks. PTX lets you write once and trust the driver to optimize for whatever hardware it lands on, whether that's Ampere, Hopper, or Blackwell.
 - You can inspect PTX output and see exactly what the compiler turned your kernel into. When you're hunting for performance, this visibility is invaluable.


---

## Why should you care about PTX?

You might never need to write PTX by hand. But understanding it pays off.

1. You can spot and fix inefficiencies that high-level CUDA hides from you. Redundant loads, suboptimal register usage, missed optimizations. You know your workload better than the compiler does.
2. You'll actually understand what your code becomes. Not abstractly, but instruction by instruction.

---

## PTX Basics

If you know CUDA, the execution model is the same: threads, warps, blocks, grids. What's new is how you access it all in PTX.

### Thread Identity

In CUDA you use `threadIdx.x`, `blockIdx.x`, and so on. In PTX, these are special registers:

```ptx
%tid.x, %tid.y, %tid.z       // thread index within block (threadIdx)
%ntid.x, %ntid.y, %ntid.z    // block dimensions (blockDim)
%ctaid.x, %ctaid.y, %ctaid.z // block index within grid (blockIdx)
%nctaid.x, %nctaid.y, %nctaid.z // grid dimensions (gridDim)
```

These are read-only and always available. No declaration needed.


### Instruction Anatomy

Before we go further, let's look at how PTX instructions are structured. Every instruction follows a pattern:

```
opcode.modifier.type  destination, source1, source2, ...
```

For example:

```ptx
add.f32  %f3, %f1, %f2;    // f3 = f1 + f2 (32-bit float)
mul.lo.s32  %r3, %r1, %r2; // r3 = low 32 bits of r1 * r2 (signed)
```

The opcode (`add`, `mul`, `ld`, `st`, etc.) tells you what operation. The modifiers refine it: `.lo` means "keep the low bits," `.f32` or `.s32` is the type. Destination comes first, then sources.

Some instructions have more modifiers. Loads and stores specify memory space and size:

```ptx
ld.global.f32  %f1, [%rd1];   // load 32-bit float from global memory
```

Here `ld` is the opcode, `.global` is the memory space, `.f32` is the type, `%f1` is the destination, and `[%rd1]` is the address to load from. The brackets mean "memory at this address," similar to pointer dereferencing.

Once you see the pattern, PTX becomes pretty readable. It's verbose compared to x86, but that verbosity makes everything explicit.

### Memory Spaces

PTX makes memory spaces explicit in every load and store. No ambiguity about where your data lives.

| Space | Prefix | Scope | Typical Use |
|-------|--------|-------|-------------|
| Global | `.global` | All threads | Main data arrays |
| Shared | `.shared` | Per-block | Inter-thread communication |
| Local | `.local` | Per-thread | Spilled registers, private arrays |
| Constant | `.const` | All threads (cached) | Read-only parameters |
| Texture | `.tex` | All threads (cached) | Spatially coherent reads |

So `ld.global.f32` loads a float from global memory, `st.shared.f32` stores to shared, and so on. The pattern is always `opcode.space.type`.

You'll see `.local` in compiler output when register pressure is high, but you rarely write to it directly.

### Registers

You declare registers upfront with a type and a count:

```ptx
.reg .b32 %r<10>;     // 10 x 32-bit untyped (bitwise ops)
.reg .u32 %ru<5>;     // 5 x 32-bit unsigned
.reg .s32 %rs<5>;     // 5 x 32-bit signed
.reg .f32 %f<8>;      // 8 x 32-bit float
.reg .f64 %fd<4>;     // 4 x 64-bit double
.reg .pred %p<3>;     // 3 x predicate (for conditionals)
```

The `<N>` syntax gives you registers numbered 0 to N-1. So `.reg .f32 %f<8>` declares `%f0` through `%f7`.

Predicates are worth calling out. PTX doesn't have traditional branching in the way you might expect. Instead, you set a predicate register and use it to guard instructions:

```ptx
setp.lt.f32 %p0, %f1, %f2;    // set %p0 if %f1 < %f2
@%p0 add.f32 %f3, %f1, %f2;   // only execute if %p0 is true
```

This is how SIMT handles divergence without full branching.

---
## Your First PTX Kernel

Let's write the classic: vector addition. Here's the CUDA version:

```cuda
__global__ void vecAdd(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```

Every PTX file starts with some metadata: the ISA version (`.version 8.0`), target architecture (`.target sm_80`), and address size (`.address_size 64`). Then you declare the kernel with `.visible .entry`, which marks it as a kernel entry point callable from host code. Device-only functions use `.func` instead.

Here's the PTX equivalent:

```ptx
.version 8.0
.target sm_80
.address_size 64

.visible .entry vecAdd(
    .param .u64 param_A,
    .param .u64 param_B,
    .param .u64 param_C,
    .param .u32 param_n
)
{
    // Declare registers
    .reg .u64 %rd<8>;       // 64-bit for pointers
    .reg .u32 %r<4>;        // 32-bit for indices
    .reg .f32 %f<3>;        // 32-bit floats for data
    .reg .pred %p0;         // Predicate for bounds check

    // Load parameters
    ld.param.u64 %rd0, [param_A];
    ld.param.u64 %rd1, [param_B];
    ld.param.u64 %rd2, [param_C];
    ld.param.u32 %r0, [param_n];

    // Calculate global thread index: i = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3;    // %r1 = i

    // Bounds check: if (i >= n) return
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra EXIT;

    // Calculate byte offsets (float = 4 bytes)
    mul.wide.u32 %rd3, %r1, 4;        // byte offset = i * 4
    add.u64 %rd4, %rd0, %rd3;         // &A[i]
    add.u64 %rd5, %rd1, %rd3;         // &B[i]
    add.u64 %rd6, %rd2, %rd3;         // &C[i]

    // Load A[i] and B[i]
    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];

    // C[i] = A[i] + B[i]
    add.f32 %f2, %f0, %f1;

    // Store result
    st.global.f32 [%rd6], %f2;

EXIT:
    ret;
}
```

The structure maps pretty directly to CUDA. Parameters come in through `.param` space, you calculate your thread index the same way, bounds check with a predicate, and do your loads/stores. More verbose, but nothing conceptually new if you followed the basics.

A few things worth noting in this example:

`mad.lo.u32` is multiply-add, keeping the low 32 bits. This is your `blockIdx.x * blockDim.x + threadIdx.x` in one instruction.

`mul.wide.u32` multiplies a 32-bit value and produces a 64-bit result. We need this because our pointers are 64-bit but our index is 32-bit.

The bounds check uses a predicate (`%p0`) and a predicated branch (`@%p0 bra EXIT`). If the thread is out of bounds, it jumps straight to the return.

## Compiling and Running PTX

You have a few options depending on what you're trying to do.

### Inline PTX Assembly

For targeted optimizations, you can embed PTX directly in CUDA code:

```cuda
int result;
asm("add.s32 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
```

The syntax follows GCC's extended asm format:

```
asm("instruction" : outputs : inputs);
```

The `%0`, `%1`, `%2` are placeholders that map to the operands in order. Constraints tell the compiler what kind of register each variable needs:

| Constraint | Meaning |
|------------|---------|
| `"r"` | 32-bit register |
| `"l"` | 64-bit register |
| `"f"` | 32-bit float register |
| `"d"` | 64-bit float register |
| `"="` | Output operand (write-only) |
| `"+"` | Output operand (read-write) |

A float example:

```cuda
float x, y, result;
asm("mul.f32 %0, %1, %2;" : "=f"(result) : "f"(x), "f"(y));
```

Use `asm volatile(...)` when the instruction has side effects the compiler shouldn't optimize away, like memory operations or barriers:

```cuda
asm volatile("bar.sync 0;");
asm volatile("cp.async.wait_group 0;");
```

This is the most practical approach for most PTX work. You keep your CUDA code, your build system, your debugging tools, and just drop to PTX for the specific instructions that matter.

Good for: specific instructions the compiler won't emit, async copy control, warp-level primitives, cache hints.

### Load PTX at Runtime

The CUDA Driver API can compile and load PTX on the fly:

```c
CUmodule module;
CUfunction kernel;

cuModuleLoadData(&module, ptx_source);  // ptx_source is a string
cuModuleGetFunction(&kernel, module, "vecAdd");

void *args[] = { &d_A, &d_B, &d_C, &n };
cuLaunchKernel(kernel, gridDim, 1, 1, blockDim, 1, 1, 0, 0, args, 0);
```

The PTX gets JIT compiled to SASS for whatever GPU is present. This is how most production systems that ship PTX actually work.

Good for: JIT compilation, runtime code generation, shipping PTX that adapts to the user's GPU architecture.

### Compile to Cubin Ahead of Time

If you want a binary for a specific architecture:

```bash
ptxas -arch=sm_80 -o kernel.cubin kernel.ptx
```

Then load it with `cuModuleLoad` instead of `cuModuleLoadData`. You can also use `ptxas` to check what SASS your PTX compiles to:

```bash
ptxas -arch=sm_80 --warn-on-spills kernel.ptx
cuobjdump -sass kernel.cubin
```

Good for: benchmarking, inspecting SASS output, distributing precompiled kernels, avoiding JIT overhead at launch.

### Which one to use?

For most optimization work: **inline asm**. You're probably not rewriting entire kernels in PTX. You're dropping in specific instructions for async copies, warp shuffles, or cache hints. Inline asm lets you do that without leaving CUDA.

For runtime flexibility: **driver API with PTX strings**. Useful when you're generating kernels dynamically or need to support multiple architectures from one binary.

For debugging codegen: **ptxas + cuobjdump**. When you want to see exactly what SASS your PTX becomes.

---

## When to Drop to PTX

vecAdd was a teaching example. You'd never actually write it in PTX by hand, the compiler does fine.

So when does PTX actually matter?

The answer is almost always: **when you need control over something the compiler won't give you.** Specific instruction selection, memory operation scheduling, or hardware features that CUDA doesn't expose cleanly.

The most common case on modern GPUs: **async memory operations**. Ampere and newer architectures have dedicated hardware for overlapping memory transfers with computation. CUDA exposes some of this through `cuda::memcpy_async` and `cuda::pipeline`, but the compiler is conservative. It won't always schedule things the way you want.

PTX gives you explicit control.

---

## Software Pipelining with Async Copies

<img src="/images/ptx/pipeline_timeline.svg" alt="Pipeline execution timeline: naive vs pipelined" style="width:100%;max-width:780px;margin:1em auto;display:block;">

Here's a real optimization pattern. Consider a kernel processing chunks of data:

```cuda
__global__ void process(float *input, float *output, int n) {
    __shared__ float tile[256];
    
    for (int i = 0; i < n; i += 256) {
        // Load phase
        tile[threadIdx.x] = input[i + threadIdx.x];
        __syncthreads();
        
        // Compute phase
        float result = expensive_compute(tile[threadIdx.x]);
        output[i + threadIdx.x] = result;
        __syncthreads();
    }
}
```

The problem is the timeline. Compute units sit idle waiting for memory:

```
Iteration 0:  [===LOAD===][===COMPUTE===]
Iteration 1:                             [===LOAD===][===COMPUTE===]
Iteration 2:                                                        [===LOAD===]...
```

Every iteration pays the full memory latency before compute can start.

The fix is software pipelining: start loading the *next* tile while computing on the *current* one. You need two buffers (double buffering) and explicit control over when loads start and when you wait for them.

PTX provides the `cp.async` family of instructions for this:

```cuda
__global__ void process_pipelined(float *input, float *output, int n) {
    __shared__ float tile[2][256];  // Double buffer
    int buf = 0;
    
    // Prime the pipeline: start loading first tile
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 4;\n"
        :: "r"(&tile[buf][threadIdx.x]), 
           "l"(&input[threadIdx.x])
    );
    asm volatile("cp.async.commit_group;\n");
    
    for (int i = 0; i < n; i += 256) {
        int next_buf = buf ^ 1;
        
        // Start async load for NEXT iteration
        if (i + 256 < n) {
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 4;\n"
                :: "r"(&tile[next_buf][threadIdx.x]), 
                   "l"(&input[i + 256 + threadIdx.x])
            );
            asm volatile("cp.async.commit_group;\n");
        }
        
        // Wait for CURRENT tile to arrive
        // wait_group 1 when next load was issued (2 groups in flight),
        // wait_group 0 on last iteration (only 1 group, must drain it)
        if (i + 256 < n) {
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();
        
        // Compute on current tile while next tile loads
        float result = expensive_compute(tile[buf][threadIdx.x]);
        output[i + threadIdx.x] = result;
        
        buf = next_buf;
    }
    
    // Drain the pipeline
    asm volatile("cp.async.wait_group 0;\n");
}
```

Now the timeline overlaps:

```
          [==LOAD 0==]
                     [==LOAD 1==][==LOAD 2==][==LOAD 3==]
                     [==COMP 0==][==COMP 1==][==COMP 2==][==COMP 3==]
```

After the first load, memory latency is hidden. Compute never stalls waiting for data.

### The PTX instructions

`cp.async.cg.shared.global [dst], [src], size` copies `size` bytes from global to shared memory asynchronously. The `.cg` modifier means "cache global" (cache in L2 only). The copy happens in the background without blocking the thread.

`cp.async.commit_group` marks the current batch of async copies as a group. You can have multiple groups in flight.

`cp.async.wait_group N` blocks until at most N groups are still pending. So `wait_group 1` means "I'm okay with one group still in flight — wait for everything else." `wait_group 0` drains everything. Be careful: if only one group is pending, `wait_group 1` returns immediately without waiting for it. That's why the code uses `wait_group 0` on the last iteration when there's no next load in flight.

### Why not just use cuda::memcpy_async?

You can. CUDA 11+ exposes `cuda::memcpy_async` and `cuda::pipeline` which compile down to these same instructions. But:

1. The compiler decides scheduling. It might not pipeline the way you want.
2. The abstraction hides what's happening. When you're debugging performance, you want to see the actual instructions.
3. PTX gives you precise control over group boundaries and wait points.

For production code, try the CUDA abstractions first. Drop to PTX when the compiler isn't giving you what you need, or when you want to understand exactly what's happening.

---

## Beyond Async Copies

Async copies are the most common entry point to PTX, but they're far from the only one. Here are more cases where PTX gives you something CUDA C++ can't.

### Explicit Prefetching to L2

Sometimes you know your access pattern better than the hardware. Global loads have ~400 cycle latency. If you can prefetch data well before you need it, you hide that latency.

```cuda
__global__ void compute_with_prefetch(float *data, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Prefetch data we'll need 4 iterations from now
    if (idx + 4 * gridDim.x * blockDim.x < n) {
        asm volatile(
            "prefetch.global.L2 [%0];\n"
            :: "l"(&data[idx + 4 * gridDim.x * blockDim.x])
        );
    }

    // Work on current data (which was prefetched 4 iterations ago)
    float val = data[idx];
    output[idx] = expensive_compute(val);
}
```

`prefetch.global.L2 [addr]` tells the memory subsystem to start fetching data into L2 cache without stalling the thread. There's no CUDA C++ device-side equivalent — `__builtin_prefetch` is a host compiler intrinsic that doesn't map to GPU prefetch instructions.

### Cache Bypass for Streaming Data

<img src="/images/ptx/cache_hierarchy.svg" alt="Cache hierarchy: default vs streaming hints" style="width:100%;max-width:780px;margin:1em auto;display:block;">

Opposite problem: you have data you'll touch exactly once. Default caching pollutes L2 with data you'll never reuse, evicting data you actually need.

```cuda
__global__ void stream_process(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val;
    // Load with cache streaming hint: minimal cache pollution
    asm volatile(
        "ld.global.cs.f32 %0, [%1];\n"
        : "=f"(val) : "l"(&input[idx])
    );

    float result = val * 2.0f;  // Simple transform

    // Store with write-through, no allocate
    asm volatile(
        "st.global.wt.f32 [%0], %1;\n"
        :: "l"(&output[idx]), "f"(result)
    );
}
```

- `ld.global.cs` — "cache streaming": load into L2 but mark for early eviction
- `st.global.wt` — "write through": write directly to memory, skip cache allocation

This can dramatically improve performance when you're bandwidth-bound and fighting cache thrashing.

### Warp-Level Reduction Without Shared Memory

The compiler generates decent shuffle code for `__shfl_down_sync`, but sometimes you need precise control — especially when combining multiple reductions or avoiding register spills.

```cuda
__device__ float warp_reduce_sum(float val) {
    asm volatile(
        "{\n"
        "  .reg .f32 temp;\n"
        "  shfl.sync.down.b32 temp, %0, 16, 0x1f, 0xffffffff;\n"
        "  add.f32 %0, %0, temp;\n"
        "  shfl.sync.down.b32 temp, %0, 8, 0x1f, 0xffffffff;\n"
        "  add.f32 %0, %0, temp;\n"
        "  shfl.sync.down.b32 temp, %0, 4, 0x1f, 0xffffffff;\n"
        "  add.f32 %0, %0, temp;\n"
        "  shfl.sync.down.b32 temp, %0, 2, 0x1f, 0xffffffff;\n"
        "  add.f32 %0, %0, temp;\n"
        "  shfl.sync.down.b32 temp, %0, 1, 0x1f, 0xffffffff;\n"
        "  add.f32 %0, %0, temp;\n"
        "}\n"
        : "+f"(val)
    );
    return val;
}
```

The scoped block with an explicit temporary register (`temp`) that the compiler can't spill, a guaranteed instruction sequence with no reordering, and the explicit `0xffffffff` mask and `0x1f` clamp are all visible and locked down.

### Tensor Core MMA — Why WMMA Isn't Enough

<img src="/images/ptx/mma_dataflow.svg" alt="MMA attention data flow: WMMA shared memory path vs PTX register path" style="width:100%;max-width:780px;margin:1em auto;display:block;">

Tensor Cores are the backbone of modern AI workloads. CUDA provides the `nvcuda::wmma` API to access them. So why would you ever need PTX?

**The WMMA workflow** looks like this:

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void matmul_wmma(half *A, half *B, float *C, float *D) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);
    wmma::load_matrix_sync(c_frag, C, 16);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    wmma::store_matrix_sync(D, c_frag, 16, wmma::mem_row_major);
}
```

This works fine for simple cases. Fragments aren't fully opaque — you *can* access individual elements via `fragment.x[i]`, and `fragment.num_elements` tells you how many each thread holds. For element-wise operations like scaling or adding a bias, this works fine:

```cuda
// Element-wise ops work — no shared memory needed
for (int i = 0; i < c_frag.num_elements; i++)
    c_frag.x[i] = alpha * c_frag.x[i] + beta;
```

But here's the catch: **the mapping of matrix elements to `fragment.x[]` indices is unspecified**. The CUDA docs explicitly state this, and NVIDIA staff have confirmed it should not be relied upon. Each thread holds `num_elements` values, but you don't know which row or column of the matrix they correspond to.

For element-wise operations, this doesn't matter — you're applying the same function to every element regardless of position. But for **row-wise operations** like softmax, it's a problem.

**Why this hurts: the softmax-matmul pattern**

In attention, you compute softmax(QK^T), then multiply by V. Softmax is *row-wise*: each row of the score matrix gets its own max and normalization. With WMMA, you don't know which elements belong to which row, so you're forced to round-trip through shared memory where the layout is known:

```cuda
// QK^T — result lands in a fragment
wmma::mma_sync(scores_frag, q_frag, k_frag, zeros_frag);

// We have fragment.x[] access, but don't know which ROW each element belongs to.
// Softmax needs row-wise max and sum, so we must go through shared memory:
wmma::store_matrix_sync(smem_scores, scores_frag, 16, wmma::mem_row_major);
__syncthreads();

// Now the layout is known — do row-wise softmax
float my_score = smem_scores[threadIdx.x];
float max_val = warp_reduce_max(my_score);
float exp_val = expf(my_score - max_val);
float sum_exp = warp_reduce_sum(exp_val);
smem_scores[threadIdx.x] = exp_val / sum_exp;
__syncthreads();

// Reload into fragment for next MMA
wmma::load_matrix_sync(scores_frag, smem_scores, 16);

// Finally multiply by V
wmma::mma_sync(output_frag, scores_frag, v_frag, zeros_frag);
```

The scores were *in registers* after the first MMA. But without knowing the row layout, softmax requires a shared memory round-trip — store, sync, softmax, sync, reload. That's 30+ cycles each way, and it gets worse when multiple warps compete for shared memory bandwidth.

**The PTX solution: registers all the way**

PTX `mma` instructions operate directly on registers, and critically, the **thread-to-element mapping is fully documented** in the PTX ISA. After an MMA, you know exactly which matrix row and column each register value corresponds to. Row-wise softmax becomes a register operation.

For `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` on sm_80, each thread in the warp holds:
- **A**: 4 registers of `.b32` (8 fp16 values packed pairwise)
- **B**: 2 registers of `.b32` (4 fp16 values packed pairwise)
- **C/D**: 4 registers of `.f32` (the accumulator input/output)

The 32 threads collectively own the full 16×8 tile. Here's how a fused attention tile looks:

```cuda
__device__ void fused_attention_tile(
    half *Q, half *K, half *V,  // Input tiles in shared memory
    float *output               // Output tile
) {
    // Register arrays for MMA operands
    unsigned A_regs[4];  // Q tile: 8 fp16 values packed into 4 u32
    unsigned B_regs[2];  // K tile: 4 fp16 values packed into 2 u32
    float C_regs[4] = {0};  // Accumulator: 4 fp32 values per thread

    // Load Q and K tiles into registers (elided for clarity)
    // ...

    // MMA: scores = Q @ K^T
    // Output lands directly in C_regs — just plain registers
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(C_regs[0]), "=f"(C_regs[1]), "=f"(C_regs[2]), "=f"(C_regs[3])
        : "r"(A_regs[0]), "r"(A_regs[1]), "r"(A_regs[2]), "r"(A_regs[3]),
          "r"(B_regs[0]), "r"(B_regs[1]),
          "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f)
    );

    // Softmax DIRECTLY on registers — no shared memory!
    float max_val = C_regs[0];
    #pragma unroll
    for (int i = 1; i < 4; i++) max_val = fmaxf(max_val, C_regs[i]);
    max_val = warp_reduce_max(max_val);  // Cross-thread max via shuffles

    float sum_exp = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        C_regs[i] = expf(C_regs[i] - max_val);
        sum_exp += C_regs[i];
    }
    sum_exp = warp_reduce_sum(sum_exp);  // Cross-thread sum via shuffles

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        C_regs[i] /= sum_exp;
    }

    // Pack softmax outputs back to fp16 for next MMA
    // The MMA output layout (D-format) doesn't directly match the A-input layout,
    // so warp shuffles are needed to rearrange registers between threads (elided)
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        half2 packed = __floats2half2_rn(C_regs[2*i], C_regs[2*i + 1]);
        A_regs[i] = *reinterpret_cast<unsigned*>(&packed);
    }
    // Remaining A_regs filled via warp shuffles to complete the layout (elided)

    // Load V tile into B_regs (elided)
    // ...

    // Second MMA: output = softmax(scores) @ V
    // Still in registers — never touched shared memory
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(C_regs[0]), "=f"(C_regs[1]), "=f"(C_regs[2]), "=f"(C_regs[3])
        : "r"(A_regs[0]), "r"(A_regs[1]), "r"(A_regs[2]), "r"(A_regs[3]),
          "r"(B_regs[0]), "r"(B_regs[1]),
          "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f)
    );

    // Store final output (elided)
}
```

**What changed:**
- After the first MMA, results are in `C_regs[]` — plain float variables
- Softmax operates directly on those registers
- Results get packed back to fp16 in registers (with warp shuffles to rearrange the layout between MMA output and input formats)
- Second MMA consumes them — no shared memory round-trip

In attention kernels, this pattern can save 100+ cycles per tile by eliminating shared memory traffic. FlashAttention and similar kernels rely on this — they use PTX `mma` (or CUTLASS's PTX wrappers), not WMMA.

**The instruction breakdown**: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
- `mma.sync` — warp-synchronous matrix multiply-accumulate
- `.aligned` — inputs are properly aligned (required)
- `.m16n8k16` — tile shape: 16 rows × 8 cols output, K=16 reduction dimension
- `.row.col` — A is row-major, B is column-major
- `.f32.f16.f16.f32` — accumulate in fp32, inputs are fp16

Each thread in the warp holds a piece of the 16×8 output tile. PTX documents exactly which registers map to which matrix elements — WMMA's `fragment.x[]` gives you the values but not the positions.

There's also a practical reason beyond layout knowledge: **WMMA doesn't expose all tile shapes**. The `m16n8k16` shape used above is PTX-only. WMMA only offers `m16n16k16`, `m32n8k16`, and `m8n32k16` for fp16. The smaller PTX tiles give you finer control over register pressure and pipeline scheduling.

<img src="/images/ptx/thread_mapping.svg" alt="Thread-to-element mapping: WMMA unknown vs PTX documented" style="width:100%;max-width:780px;margin:1.5em auto;display:block;">

**Hopper's wgmma**

On H100, NVIDIA introduced `wgmma` — warp-group MMA operating on 128 threads with tiles up to 64×256×16. There's no WMMA API for this. PTX is the only way to access it:

```cuda
asm volatile(
    "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
    "{%0, %1, ...}, desc_a, desc_b, scale_d, scale_a, scale_b;\n"
    : /* 128+ output registers */
    : /* TMA descriptors */
);
```

If you're building inference engines targeting H100, you'll be writing PTX.

### How big is the gap?

We benchmarked the WMMA shared-memory path against PTX register-only softmax on an RTX 4090, sweeping tile counts from 1K to 512K with varying warps per block. The PTX path is **2.5–3.7x faster** consistently — and the gap widens as the GPU saturates:

<img src="/images/ptx/mma_scaling.svg" alt="MMA latency: WMMA vs PTX across tile counts" style="width:100%;max-width:700px;margin:1.5em auto;display:block;">

The throughput picture makes it even clearer — PTX sustains over 2 billion attention tiles/sec while WMMA plateaus under 800M:

<img src="/images/ptx/mma_throughput.svg" alt="Attention throughput: WMMA vs PTX" style="width:100%;max-width:700px;margin:1.5em auto;display:block;">

When multiple warps share a block, the WMMA version competes for shared memory bandwidth. The PTX version works entirely in registers — no contention:

<img src="/images/ptx/mma_contention.svg" alt="Shared memory contention across warps" style="width:100%;max-width:500px;margin:1.5em auto;display:block;">

### Forcing Register Allocation

Sometimes you know exactly how many registers a hot loop needs and want to prevent spills:

```cuda
__global__ __launch_bounds__(256, 4)  // 256 threads, aim for 4 blocks/SM
void register_controlled_kernel(float *data) {
    float a, b, c, d;
    asm volatile(
        "{\n"
        "  .reg .f32 r0, r1, r2, r3;\n"
        "  ld.global.f32 r0, [%4];\n"
        "  ld.global.f32 r1, [%4+4];\n"
        "  ld.global.f32 r2, [%4+8];\n"
        "  ld.global.f32 r3, [%4+12];\n"
        "  mul.f32 r0, r0, r1;\n"
        "  fma.rn.f32 r0, r2, r3, r0;\n"
        "  mov.f32 %0, r0;\n"
        "  mov.f32 %1, r1;\n"
        "  mov.f32 %2, r2;\n"
        "  mov.f32 %3, r3;\n"
        "}\n"
        : "=f"(a), "=f"(b), "=f"(c), "=f"(d)
        : "l"(data + threadIdx.x * 4)
    );
}
```

The scoped block `{ }` with explicit `.reg` declarations tells `ptxas` exactly what you need. Combined with `__launch_bounds__`, this gives you fine-grained occupancy control.

---

## When NOT to Use PTX

- **The compiler does fine** — check `nvcc -ptx` output first. Often it's already optimal.
- **Readability matters more** — PTX is hard to maintain. Future-you will curse present-you.
- **Portability matters** — some PTX features are architecture-specific. Your sm_80 code might not run on sm_90.
- **You're guessing** — profile first. PTX micro-optimizations rarely beat algorithmic improvements.

Use PTX for the innermost 10% of your kernel that runs 90% of the time, after you've verified the compiler isn't already doing what you need.

---

## Further Reading

- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/) — The definitive 400+ page spec
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/) — `cuobjdump`, `nvdisasm` for inspecting binaries
- [Inline PTX Assembly in CUDA](https://docs.nvidia.com/cuda/inline-ptx-assembly/)

---

## Benchmarks

All benchmarks run on an RTX 4090 (sm_89, CUDA 12.0). Not every PTX optimization yields a dramatic speedup — the 4090's massive L2 cache and memory bandwidth already hide a lot of latency. The wins are specific: tensor core control is where PTX earns its keep.

<img src="/images/ptx/benchmark_overview.svg" alt="PTX benchmark overview: speedup across all techniques" style="width:100%;max-width:700px;margin:1.5em auto;display:block;">

| Technique | Baseline | PTX | Speedup |
|-----------|----------|-----|---------|
| Software Pipelining (cp.async) | 0.437 ms | 0.422 ms | 1.03x |
| Cache Hints (ld.cs / st.wt) | 13.9 ms | 14.0 ms | ~1.0x |
| Warp Reduction (shfl) | 1.790 ms | 1.786 ms | ~1.0x |
| **MMA Attention** | **0.328 ms** | **0.120 ms** | **2.7x** |

The pipelining and cache results are honest: on a GPU this powerful, the compiler and hardware already do a good job. The shuffle reduction confirms the blog's point — the compiler generates near-identical SASS, so PTX shfl is about guaranteed instruction ordering, not raw speed.

The MMA result is the real story. Eliminating the shared memory round-trip for row-wise softmax is a **2.7x** win that scales consistently across workload sizes. This is why FlashAttention and CUTLASS use PTX.

*Benchmark code: [github.com/dhmnr/ptx-bench](https://github.com/dhmnr/ptx-bench)*

---

## Conclusion

We covered PTX from scratch: what instructions look like, how memory and registers work, how to write and run a kernel. Then we looked at where it actually matters — async copies for software pipelining, cache control, warp shuffles, and direct Tensor Core access via `mma` with a documented thread-to-element mapping that WMMA doesn't provide.

The MMA example is where this gets real. WMMA's `fragment.x[]` gives you element access, but without knowing which row each element belongs to, row-wise operations like softmax require a shared memory round-trip. PTX's `mma` documents the exact layout, keeping everything in registers. That's a 2.7x gap on real hardware — and it's the reason production attention kernels use PTX, not WMMA.

In Part 2, we'll go deeper: multi-stage pipelining with more than two buffers, TMA on Hopper, and building a complete attention tile from scratch in PTX.

---

*Questions? Found an error? Let me know in the comments.*
