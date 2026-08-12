<table>
<tr>
<td valign="top">

# GEMM (General Matrix Multiply)

Tiled, shared-memory matrix multiplication, in two versions: one assuming
the matrix width divides evenly into the tile size, and one generalized to
handle widths that don't.

</td>
<td valign="top" align="right" width="320">

**Skills covered**

- Shared-memory tiling for reuse
- Boundary handling for non-divisible matrix widths
- Zero-padding out-of-range shared-memory loads
- Thread/block indexing for 2D tiled workloads

</td>
</tr>
</table>

## Table of contents

- [What is tiled matrix multiplication](#what-is-tiled-matrix-multiplication)
- [Implementations](#implementations)
- [Compiling and running](#compiling-and-running)

## What is tiled matrix multiplication

Naive matrix multiplication computes each output element `P[row][col]` as a
dot product over a full row of `M` and a full column of `N`, reading both
directly from global memory for every multiply-add. Neighboring threads
computing neighboring output elements re-read much of the same row/column
data from global memory, over and over.

Tiling fixes this by having each thread block cooperatively load a small
`TILE_WIDTH x TILE_WIDTH` square of `M` and `N` into `__shared__` memory
once, then having every thread in the block reuse that shared tile for
several multiply-adds before moving to the next tile. This trades repeated
global memory reads for a handful of fast shared memory reads per tile,
which is the same idea used by `conv_2d_shared` in
[../Convolution](../Convolution).

## Implementations

### `tileMatMul.cu`: divisible widths

Assumes `width` is evenly divisible by `TILE_WIDTH`. Each thread walks
`width / TILE_WIDTH` tiles, loading one element of `M` and one element of
`N` into shared memory per tile, synchronizing, accumulating
`TILE_WIDTH` partial products, synchronizing again, then moving to the next
tile:

```cpp
for (int phase = 0; phase < width / TILE_WIDTH; ++phase)
{
    ds_M[threadIdx.y][threadIdx.x] = m[row * width + (threadIdx.x + TILE_WIDTH * phase)];
    ds_N[threadIdx.y][threadIdx.x] = n[(threadIdx.y + phase * TILE_WIDTH) * width + col];
    __syncthreads();

    for (int i = 0; i < TILE_WIDTH; i++)
        pVal += ds_M[threadIdx.y][i] * ds_N[i][threadIdx.x];
    __syncthreads();
}
```

No bounds checks are needed on the loads or the final write, since
`width / TILE_WIDTH` tiles exactly cover the matrix with no partial tile at
the edge.

### `tileMatMulGeneric.cu`: arbitrary widths

Same tiling idea, generalized to widths that don't divide evenly into
`TILE_WIDTH`. Two changes make that possible:

- **Tile count rounds up.** `width / TILE_WIDTH` becomes
  `(width - 1) / TILE_WIDTH + 1`, so a partial tile at the edge still gets
  processed instead of being dropped. The grid is sized to match:
  `(width + TILE_WIDTH - 1) / TILE_WIDTH` blocks per dimension instead of
  `width / TILE_WIDTH`.
- **Loads and the final write are bounds-checked.** In the last, partial
  tile, some threads would otherwise read past the edge of `M` or `N`.
  Each load checks whether its source index is still inside the matrix,
  and writes zero into shared memory instead of reading out of bounds when
  it isn't:

  ```cpp
  if (row < width && phase * TILE_WIDTH + threadIdx.x < width)
      ds_M[threadIdx.y][threadIdx.x] = m[row * width + (threadIdx.x + TILE_WIDTH * phase)];
  else
      ds_M[threadIdx.y][threadIdx.x] = 0.0;
  ```

  The same guard shows up before the final write to `p[row * width + col]`,
  since some threads in edge blocks don't correspond to a real output
  element at all.

## Compiling and running

```powershell
nvcc tileMatMul.cu -o tileMatMul.exe
.\tileMatMul.exe

nvcc tileMatMulGeneric.cu -o tileMatMulGeneric.exe
.\tileMatMulGeneric.exe
```

Both programs generate a random `width x width` input on the host, run the
kernel once, and free their buffers; neither prints the result or times the
kernel yet.
