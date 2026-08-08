# CUDA-Performance-Lab

A collection of CUDA kernels written while learning GPU programming, profiling,
and performance optimization. Each program pairs a working kernel with
measured benchmarks and, where relevant, Nsight Systems/Compute profiling.

## Table of contents

- [Test environment](#test-environment)
- [Programs](#programs)
- [Compiling and running](#compiling-and-running)

## Test environment

All benchmarks and profiling numbers in this repo (including sub-folder
READMEs) were measured on:

- **GPU:** NVIDIA GeForce GTX 1650, 14 SMs
- **Peak memory bandwidth:** ~192 GB/s

## Programs

### Complete and runnable

- **[Convolution/](Convolution/)**. 1D and 2D mask-centered convolution, each
  with a naive (global-memory) baseline and a shared-memory tiled variant
  (`conv_1d`/`conv_1d_shared`, `conv_2d`/`conv_2d_shared`). The 2D tiled
  kernel also uses `__constant__` memory for the mask and `cudaMallocPitch`
  for its buffers. Benchmarked with `cudaEvent`s and profiled with both
  Nsight Systems and Nsight Compute. Full writeup in
  [Convolution/README.md](Convolution/README.md).

- **[histogram/](histogram/)**. Counts occurrences of `a-z` in a byte
  buffer, built up as a sequence of measured optimizations: naive global
  `atomicAdd`, block-level privatization, then register-level privatization,
  plus a launch config sized off the GPU rather than the input. Privatization
  measured ~2.9x faster than the naive baseline once the grid was sized
  correctly. Full writeup, numbers, and timeline screenshot in
  [histogram/README.md](histogram/README.md).

- **tileMatMul.cu**. Tiled square matrix multiplication using shared memory.
  Loads `TILE_WIDTH x TILE_WIDTH` tiles of the input matrices into shared
  memory per block to reduce global memory traffic. Assumes the matrix width
  is evenly divisible by `TILE_WIDTH`.

- **tileMatMulGeneric.cu**. Same tiled matrix multiplication approach as
  `tileMatMul.cu`, generalized to matrix widths that are *not* evenly
  divisible by `TILE_WIDTH`, with boundary checks that zero-pad out-of-range
  shared memory loads.

### Kernel only, host setup pending

- **clr_greyscale.cu**. Converts an RGB image to grayscale. Each thread maps
  to one pixel, reading the 3-channel RGB value and writing a single
  grayscale value using the standard luminance weights
  (0.21R + 0.71G + 0.07B). `main()` is a kernel-launch sketch referencing
  undeclared variables (`n`, `m`, `w`, `h`, `d_Pin`, `d_Pout`); host-side
  allocation and launch setup haven't been written yet.

- **blurKernel.cu**. Applies a simple box blur to a single-channel image.
  Each thread averages a `(2*BLUR_SIZE+1) x (2*BLUR_SIZE+1)` neighborhood
  around its pixel, clamping at image boundaries. `main()` is a
  kernel-launch sketch referencing undeclared variables (`n`, `m`, `in`,
  `out`, `width`, `height`); host-side setup hasn't been written yet.

## Compiling and running

Requires the NVIDIA CUDA Toolkit (`nvcc`) and a CUDA-capable GPU.

**Convolution** (from the `Convolution/` folder):

```powershell
nvcc convolution.cu -o convolution.exe
.\convolution.exe
```

**tileMatMul / tileMatMulGeneric**:

```powershell
nvcc tileMatMul.cu -o tileMatMul.exe
.\tileMatMul.exe

nvcc tileMatMulGeneric.cu -o tileMatMulGeneric.exe
.\tileMatMulGeneric.exe
```

**histogram** (from the `histogram/` folder) needs an extra flag plus a test
file, since it reads input at runtime rather than generating it:

```powershell
cd histogram
nvcc -std=c++17 histogram.cu -o histogram.exe
.\histogram.exe
```

The `-std=c++17` flag is required; CUDA's `<cuda/atomic>` header won't
compile without it. `input.txt` isn't checked into the repo (see
[histogram/README.md](histogram/README.md) for why); generate your own
before running, e.g. a few MB of random bytes.

`clr_greyscale.cu` and `blurKernel.cu` aren't runnable yet; they'll get the
same treatment once their host-side code lands.
