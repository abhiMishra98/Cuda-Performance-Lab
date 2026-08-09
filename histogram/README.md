# Histogram

Counts occurrences of `a-z` in a byte buffer, bucketed into 7 groups of 4
letters each (`[a-d]`, `[e-h]`, ... `[y-z]`). Built up as a sequence of
measured optimizations rather than a single kernel.

## Table of contents

- [What is a histogram kernel doing here](#what-is-a-histogram-kernel-doing-here)
- [Implementations](#implementations)
- [Coalesced memory access](#coalesced-memory-access)
- [Atomic operations](#atomic-operations)
- [Launch config: sizing the grid to the GPU, not the input](#launch-config-sizing-the-grid-to-the-gpu-not-the-input)
- [Measured results (Nsight Systems)](#measured-results-nsight-systems)
- [Next optimization: overlapping transfer with compute (planned)](#next-optimization-overlapping-transfer-with-compute-planned)

## What is a histogram kernel doing here

A histogram counts how many times each value (or bucket of values) occurs
in a dataset. On a GPU, the natural approach, every thread reading its own
input and incrementing a shared counter array, runs straight into a
correctness problem: many threads increment the *same* bucket at the same
time. That makes histogramming a good small case study for two things this
folder walks through: coalesced memory access, and safely handling
concurrent updates to shared state.

## Implementations

All kernels live in [histogram.cu](histogram.cu).

### `histo_kernel_naive`: global atomics

Every thread walks its slice of the input and calls `atomicAdd` directly on
the global histogram array for every byte it processes. Correct, but every
single increment contends for global memory, which has the highest latency
of any GPU memory space.

### `histo_kernel`: block + register privatization

Two levels of private, low-latency counters are layered in front of the
global array before any atomic touches global memory. Covered in detail
below.

## Coalesced memory access

Each thread starts at `tIdx = blockIdx.x * blockDim.x + threadIdx.x` and
advances by `stride = blockDim.x * gridDim.x` per iteration. At every step,
consecutive threads (`threadIdx.x`, `threadIdx.x + 1`, ...) read consecutive
bytes (`buffer[tIdx]`, `buffer[tIdx + 1]`, ...). The GPU can merge these
per-thread reads into one wide memory transaction instead of issuing a
separate transaction per thread. This merging is coalescing, and it's why
threads are indexed to walk the buffer in lockstep rather than each owning
a separate contiguous chunk.

## Atomic operations

**What they are.** A hardware-guaranteed way to perform a read-modify-write
as a single, indivisible step, so that concurrent updates from different
threads can't interleave and drop each other's results.

**Why they're needed here.** Multiple threads incrementing the same
histogram bin is a classic read-modify-write hazard. Without atomics,
`d_histoBuff[bin]++` compiles to a separate load, add, and store. If two
threads target the same bin, their loads/adds/stores can interleave in a
way that drops an update: both read the old count, both add 1, and both
write back the same new value, so one increment is silently lost. With 256
threads per block and only 7 bins, collisions on the same bin are frequent
here, not an edge case. `atomicAdd` makes that read-modify-write
indivisible at the hardware level, so concurrent increments to the same
address always accumulate correctly.

**How they're used in this implementation, and why two extra layers sit in
front of them.** Calling `atomicAdd` on the global histogram for every byte
(as `histo_kernel_naive` does) works, but every one of those atomics
contends for the same small set of global addresses. `histo_kernel` reduces
that contention in two steps:

- **Block-level privatization.** Instead of every thread atomically
  updating one histogram in global memory, each block keeps its own copy in
  shared memory (`d_histoBuff[7]`). Threads within a block still contend
  with each other via atomics on this shared copy, but blocks no longer
  contend with *each other* over global memory, which is far more
  expensive since global memory has much higher latency than shared
  memory. After each block finishes accumulating locally, it does one
  atomic merge per bin into the global `histo` array. This is why the
  kernel synchronizes twice: once after zeroing the shared buffer, and
  once after all threads finish accumulating into it, before the final
  merge.

- **Register-level privatization.** Block-level privatization still leaves
  every thread in a block contending for the same 7 addresses in shared
  memory; with 256 threads per block, that's still frequent collisions on
  `atomicAdd(&d_histoBuff[bin], ...)`. Register privatization adds one more
  level underneath: each thread keeps its own `reg_histoBuff[7]` in
  registers and, while walking its grid-stride loop, increments its
  private copy with a plain `reg_histoBuff[bin]++`, no atomics, no
  contention, since no other thread can see or touch another thread's
  registers. Only after the loop finishes does each thread merge its 7
  private counts into the shared block histogram, via one `atomicAdd` per
  non-zero bin. This turns atomic traffic from one atomic per byte
  processed into at most 7 atomics per thread, total, regardless of how
  much of the buffer that thread walked.

## Launch config: sizing the grid to the GPU, not the input

The naive way to size a grid-stride kernel is
`gridDim = (size + blockDim - 1) / blockDim`, just enough blocks to cover
`size` elements in one pass. This is a trap for privatization: at large
`size`, it produces roughly one thread per element, so the grid-stride
`while` loop only ever runs once (or zero times) per thread. Register
privatization's entire benefit comes from amortizing many increments into
one atomic merge; with one element per thread, there's nothing to amortize,
so the kernel pays privatization's fixed costs (shared-memory zeroing, two
`__syncthreads()`, a 7-element merge loop per thread) for zero benefit. This
was measured directly: on a 100MB input with a size-scaled grid, the naive
kernel (6.02ms) beat the privatized one (9.58ms).

The fix is to decouple grid size from input size entirely and size it off
the GPU instead, so every thread strides across many elements regardless of
how large the input gets:

```cpp
int numBlocksPerSm = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, histo_kernel, blockDim.x, 0);
dim3 gridDim(deviceProp.multiProcessorCount * numBlocksPerSm);
```

`cudaOccupancyMaxActiveBlocksPerMultiprocessor` reports how many blocks of
*this exact kernel*, given its actual register and shared-memory usage, can
be simultaneously resident on one SM. Multiplying by the SM count gives a
grid that keeps every SM fully occupied without over-launching. On the GPU
used for this benchmark (see the [repo root README](../README.md#test-environment)
for specs), that resolved to 4 blocks/SM, 56 blocks total, launched
regardless of whether the input is 10KB or 1GB.

## Measured results (Nsight Systems)

10MB test input on the launch config above:

| Kernel | Total time | Notes |
|---|---|---|
| `histo_kernel_naive` (plain global `atomicAdd`, no privatization) | 6.05 ms | every hit contends on global memory directly |
| `histo_kernel` (block + register privatization) | 2.09 ms | atomics only at the per-block merge step |

**~2.9x faster** with privatization, once the grid is sized correctly. Both
kernels were verified to produce identical bucket counts on the same input
before comparing timing.

Captured with:

```powershell
nsys profile --trace=cuda,nvtx,cublas,cuDNN --output=histogram_profile histogram.exe
nsys stats --report cuda_gpu_kern_sum histogram_profile.nsys-rep
```

![Nsight Systems timeline showing the H2D transfer fully blocking before either kernel starts](../images/histogram_V1_NsightSystems.jpg)

Zooming into the transfer window shown above: under `CUDA HW`, the
`Memcpy HtoD (Pageable)` bar runs from ~391.5ms to ~394ms, and *both* kernel
sub-rows (`histo_kernel_naive`, `histo_kernel`) are completely empty for
that entire span; neither kernel starts until the transfer has fully
finished. On the host side, `cudaMemcpy` blocks the CPU thread for the same
duration, so nothing useful happens concurrently on either the GPU or the
CPU during that ~2.5ms window.

## Next optimization: overlapping transfer with compute (planned)

The current version copies the entire input to the device with one
synchronous, pageable `cudaMemcpy` before either kernel launches; the
screenshot above is the direct evidence that transfer and compute never
overlap. The planned fix:

1. Allocate the host buffer with `cudaMallocHost` instead of `malloc`.
   Pinned (page-locked) memory is a hard requirement for `cudaMemcpyAsync`
   to actually run asynchronously; on pageable memory it silently falls
   back to synchronous behavior.
2. Split the input into chunks and create one `cudaStream_t` per chunk.
3. For each chunk, issue `cudaMemcpyAsync(..., stream[i])` followed by
   `histo_kernel<<<..., stream[i]>>>(...)` on that same stream, all writing
   into the same `d_histo` via `atomicAdd`. This is safe even with several
   chunks' kernels running concurrently, since atomics are safe across
   streams, not just across blocks within one kernel.

Success criterion: in a re-profiled `CUDA HW` view, a chunk's
`Memcpy HtoD` bar and a *different* chunk's kernel bar should overlap
horizontally, instead of the kernel rows sitting empty under the transfer
bar as they do above.
