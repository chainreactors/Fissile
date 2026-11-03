"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Exact selected functions from pinned FlashInfer utils.py; no arithmetic changes.
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op

def ceil_div(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


@dsl_user_op
def sm120_make_smem_layout_sfa(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """Make shared memory layout for scale factors A on SM120.

    Constructs the SMEM layout for SFA based on BlockScaledBasicChunk,
    MMA tiler shape, scale factor vector size, and number of pipeline stages.

    Args:
        tiled_mma: The tiled MMA operation.
        tile_shape_mnk: The tile shape (M, N, K).
        sf_vec_size: Scale factor vector size (16 or 32).
        num_stages: Number of pipeline stages.

    Returns:
        Staged shared memory layout for SFA.
    """
    assert sf_vec_size == 16 or sf_vec_size == 32, "sf_vec_size must be 16 or 32"

    blk_mn = 128
    blk_sf = 4
    blk_elems = blk_mn * blk_sf
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size

    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (16, 4)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)

    # Sub-64-row M tiles appear only in the b12x dense GEMM path, whose SF
    # fragment layouts handle them.
    assert tile_shape_mnk[0] % (blk_mn // 8) == 0, (
        "tile_shape_mnk[0] must be divisible by 16"
    )

    # Scale-factor tiles are quantized in 128-row blocks, so narrower MMA
    # tiles still allocate one full SF block and consume only the live subset.
    sfa_tile_m = max(blk_mn, ceil_div(tile_shape_mnk[0], blk_mn) * blk_mn)

    sSFA_shapeM = (mn_basic_block_shape, sfa_tile_m // blk_mn)
    sSF_strideM = (mn_basic_block_stride, blk_elems)

    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )
    assert tile_shape_mnk[2] % (sf_vec_size * blk_sf) == 0, (
        "tile_shape_mnk[2] must be divisible by sf_vec_size * blk_sf"
    )
    assert blk_sf % mma_nsf == 0, "blk_sf must be divisible by mma_nsf"

    sSFA_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sSF_strideK = (
        k_basic_block_stride,
        mma_nsf,
        sfa_tile_m // blk_mn * blk_elems,
    )

    sSFA_shape = (sSFA_shapeM, sSFA_shapeK)
    sSFA_stride = (sSF_strideM, sSF_strideK)

    smem_layout = cute.make_layout(sSFA_shape, stride=sSFA_stride)

    sfa_smem_layout_staged = cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )

    return sfa_smem_layout_staged


@dsl_user_op
def sm120_make_smem_layout_sfb(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """Make shared memory layout for scale factors B on SM120.

    Constructs the SMEM layout for SFB based on BlockScaledBasicChunk,
    MMA tiler shape, scale factor vector size, and number of pipeline stages.

    Args:
        tiled_mma: The tiled MMA operation.
        tile_shape_mnk: The tile shape (M, N, K).
        sf_vec_size: Scale factor vector size (16 or 32).
        num_stages: Number of pipeline stages.

    Returns:
        Staged shared memory layout for SFB.
    """
    blk_mn = 128
    blk_sf = 4
    blk_elems = blk_mn * blk_sf

    assert sf_vec_size == 16 or sf_vec_size == 32, "sf_vec_size must be 16 or 32"

    assert tile_shape_mnk[1] % (blk_mn // 8) == 0, (
        "tile_shape_mnk[1] must be divisible by 16"
    )

    assert tile_shape_mnk[2] % sf_vec_size == 0, (
        "tile_shape_mnk[2] must be divisible by sf_vec_size"
    )

    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size

    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (16, 4)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)

    # Scale-factor tiles are quantized in 128-column blocks, so narrower MMA
    # tiles still allocate one full SF block and consume only the live subset.
    sfb_tile_n = max(blk_mn, ceil_div(tile_shape_mnk[1], blk_mn) * blk_mn)

    sSFB_shapeN = (mn_basic_block_shape, sfb_tile_n // blk_mn)
    sSF_strideN = (mn_basic_block_stride, blk_elems)

    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )
    assert tile_shape_mnk[2] % (sf_vec_size * blk_sf) == 0, (
        "tile_shape_mnk[2] must be divisible by sf_vec_size * blk_sf"
    )
    assert blk_sf % mma_nsf == 0, "blk_sf must be divisible by mma_nsf"

    sSFB_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sSF_strideK = (
        k_basic_block_stride,
        mma_nsf,
        sfb_tile_n // blk_mn * blk_elems,
    )

    sSFB_shape = (sSFB_shapeN, sSFB_shapeK)
    sSFB_stride = (sSF_strideN, sSF_strideK)

    smem_layout = cute.make_layout(sSFB_shape, stride=sSFB_stride)

    sfb_smem_layout_staged = cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )

    return sfb_smem_layout_staged

