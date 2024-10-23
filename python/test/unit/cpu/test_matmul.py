import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def prepack_kernel(in_p, out_p, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                   BLOCK_SIZE_N: tl.constexpr, BLOCKED_OUTPUT: tl.constexpr, TRANSPOSE: tl.constexpr,
                   PACK32: tl.constexpr):
    B_PACK_SCALE: tl.constexpr = 32 // in_p.type.element_ty.primitive_bitwidth if PACK32 else 1
    tl.static_assert(B_PACK_SCALE >= 1 and B_PACK_SCALE <= 4)
    tl.static_assert(M % BLOCK_SIZE_M == 0)
    tl.static_assert(N % BLOCK_SIZE_N == 0)
    tl.static_assert(BLOCK_SIZE_M % B_PACK_SCALE == 0)
    tl.static_assert(BLOCKED_OUTPUT or not TRANSPOSE)
    tl.static_assert(BLOCKED_OUTPUT or PACK32)
    in_block_m = tl.program_id(0)
    in_block_n = tl.program_id(1)
    out_block_m = in_block_n if TRANSPOSE else in_block_m
    out_block_n = in_block_m if TRANSPOSE else in_block_n
    BLOCK_IN_OFFS = in_block_m * BLOCK_SIZE_M * N + in_block_n * BLOCK_SIZE_N
    OUT_STRIDE_M: tl.constexpr = BLOCK_SIZE_N * B_PACK_SCALE if BLOCKED_OUTPUT else N * B_PACK_SCALE
    OUT_STRIDE_BLOCK_N: tl.constexpr = BLOCK_SIZE_M * BLOCK_SIZE_N if BLOCKED_OUTPUT else BLOCK_SIZE_N * B_PACK_SCALE
    OUT_STRIDE_BLOCK_M: tl.constexpr = M * BLOCK_SIZE_N if TRANSPOSE else BLOCK_SIZE_M * N
    BLOCK_OUT_OFFS = out_block_m * OUT_STRIDE_BLOCK_M + out_block_n * OUT_STRIDE_BLOCK_N
    for i in tl.range(0, BLOCK_SIZE_M // B_PACK_SCALE):
        row1 = tl.load(in_p + BLOCK_IN_OFFS + N * i * B_PACK_SCALE + tl.arange(0, BLOCK_SIZE_N))
        if B_PACK_SCALE > 1:
            row2 = tl.load(in_p + BLOCK_IN_OFFS + N * (i * B_PACK_SCALE + 1) + tl.arange(0, BLOCK_SIZE_N))
            if B_PACK_SCALE > 2:
                row3 = tl.load(in_p + BLOCK_IN_OFFS + N * (i * B_PACK_SCALE + 2) + tl.arange(0, BLOCK_SIZE_N))
                row4 = tl.load(in_p + BLOCK_IN_OFFS + N * (i * B_PACK_SCALE + 3) + tl.arange(0, BLOCK_SIZE_N))
                row1 = tl.ravel(tl.join(row1, row3))
                row2 = tl.ravel(tl.join(row2, row4))
            row1 = tl.ravel(tl.join(row1, row2))
        tl.store(out_p + BLOCK_OUT_OFFS + OUT_STRIDE_M * i + tl.arange(0, BLOCK_SIZE_N * B_PACK_SCALE), row1)


@triton.jit
def matmul_kernel_amx(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                      BLOCK_SIZE_K: tl.constexpr,
                      # number of blocks in a group
                      GROUP_SIZE_M: tl.constexpr, GROUP_SIZE_N: tl.constexpr, BLOCKED_A: tl.constexpr,
                      BLOCKED_B: tl.constexpr, TRANSPOSED_B: tl.constexpr, PACKED_B: tl.constexpr):
    pid = tl.program_id(axis=0)
    group_id = pid // (GROUP_SIZE_M * GROUP_SIZE_N)
    groups_n = N // BLOCK_SIZE_N // GROUP_SIZE_N
    group_m = group_id // groups_n
    group_n = group_id % groups_n
    block_id = pid % (GROUP_SIZE_M * GROUP_SIZE_N)
    block_m = group_m * GROUP_SIZE_M + block_id // GROUP_SIZE_N
    block_n = group_n * GROUP_SIZE_N + block_id % GROUP_SIZE_N

    a_stride_k = 1
    a_stride_m = BLOCK_SIZE_K if BLOCKED_A else K
    a_stride_block_k = BLOCK_SIZE_M * BLOCK_SIZE_K if BLOCKED_A else BLOCK_SIZE_K
    a_stride_block_m = BLOCK_SIZE_M * K

    B_PACK_SCALE: tl.constexpr = 32 // b_ptr.type.element_ty.primitive_bitwidth if PACKED_B else 1
    PACKED_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_K // B_PACK_SCALE if PACKED_B else BLOCK_SIZE_K
    PACKED_BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_N * B_PACK_SCALE if PACKED_B else BLOCK_SIZE_N
    assert BLOCKED_B or not TRANSPOSED_B
    b_stride_n = 1
    b_stride_k = PACKED_BLOCK_SIZE_N if BLOCKED_B else N * B_PACK_SCALE
    if TRANSPOSED_B:
        b_stride_block_n = BLOCK_SIZE_N * K
        b_stride_block_k = BLOCK_SIZE_K * BLOCK_SIZE_N
    else:
        b_stride_block_n = BLOCK_SIZE_K * BLOCK_SIZE_N if BLOCKED_B else PACKED_BLOCK_SIZE_N
        b_stride_block_k = BLOCK_SIZE_K * N

    a_block_ptr = tl.make_block_ptr(base=a_ptr,
                                    shape=(M // BLOCK_SIZE_M, K // BLOCK_SIZE_K, BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    strides=(a_stride_block_m, a_stride_block_k, a_stride_m, a_stride_k),
                                    offsets=(block_m, 0, 0, 0), block_shape=(1, 1, BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(3, 2, 1, 0))
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K // BLOCK_SIZE_K, N // BLOCK_SIZE_N, PACKED_BLOCK_SIZE_K, PACKED_BLOCK_SIZE_N),
        strides=(b_stride_block_k, b_stride_block_n, b_stride_k, b_stride_n), offsets=(0, block_n, 0, 0),
        block_shape=(1, 1, PACKED_BLOCK_SIZE_K, PACKED_BLOCK_SIZE_N), order=(3, 2, 1, 0))
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(N, 1),
                                    offsets=(block_m * BLOCK_SIZE_M, block_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_block_ptr).reshape((BLOCK_SIZE_M, BLOCK_SIZE_K))
        b = tl.load(b_block_ptr).reshape((PACKED_BLOCK_SIZE_K, PACKED_BLOCK_SIZE_N))

        c += tl.dot(a, b, out_dtype=tl.float32, rhs_encoding="row_major_interleaved" if PACKED_B else "row_major")

        a_block_ptr = tl.advance(a_block_ptr, (0, 1, 0, 0))
        b_block_ptr = tl.advance(b_block_ptr, (1, 0, 0, 0))

    tl.store(c_block_ptr, c)


@pytest.mark.parametrize("M, N, K",
                         [(m, n, k) for m in (128, 256, 512) for n in (128, 256, 512) for k in (128, 256, 512)])
@pytest.mark.parametrize("lhs_dtype, rhs_dtype, res_dtype", [('bfloat16', 'bfloat16', 'float32'),
                                                             ('float16', 'float16', 'float32')])
@pytest.mark.parametrize("BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K",
                         [(m, n, k) for m in (16, 32) for n in (16, 32) for k in (32, 64)])
@pytest.mark.parametrize("GROUP_SIZE_M, GROUP_SIZE_N", [(1, 1), (2, 2), (2, 4), (4, 2), (4, 4)])
@pytest.mark.parametrize("BLOCKED_A", [False, True])
@pytest.mark.parametrize("BLOCKED_B, TRANSPOSED_B", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("PACKED_B", [False, True])
def test_matmul(M, N, K, lhs_dtype, rhs_dtype, res_dtype, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
                GROUP_SIZE_N, BLOCKED_A, BLOCKED_B, TRANSPOSED_B, PACKED_B, device):
    assert M % (GROUP_SIZE_M * BLOCK_SIZE_M) == 0, f"M={M}, GROUP_SIZE_M={GROUP_SIZE_M}, BLOCK_SIZE_M={BLOCK_SIZE_M}"
    assert N % (GROUP_SIZE_N * BLOCK_SIZE_N) == 0, f"N={N}, GROUP_SIZE_N={GROUP_SIZE_N}, BLOCK_SIZE_N={BLOCK_SIZE_N}"
    assert K % BLOCK_SIZE_K == 0, f"K={K}, BLOCK_SIZE_K={BLOCK_SIZE_K}"
    assert BLOCKED_B or not TRANSPOSED_B

    a = torch.randn((M, K), device=device, dtype=getattr(torch, lhs_dtype))
    b = torch.randn((K, N), device=device, dtype=getattr(torch, rhs_dtype))
    c = torch.zeros((M, N), device=device, dtype=getattr(torch, res_dtype))

    ref = torch.matmul(a.to(c.dtype), b.to(c.dtype))

    if BLOCKED_A:
        ab = torch.empty_like(a)
        prepack_kernel[(M // BLOCK_SIZE_M, K // BLOCK_SIZE_K)](a, ab, M, K, BLOCK_SIZE_M=BLOCK_SIZE_M,
                                                               BLOCK_SIZE_N=BLOCK_SIZE_K, BLOCKED_OUTPUT=True,
                                                               TRANSPOSE=False, PACK32=False)
        a = ab

    if BLOCKED_B or PACKED_B:
        bb = torch.empty_like(b)
        prepack_kernel[(K // BLOCK_SIZE_K, N // BLOCK_SIZE_N)](b, bb, K, N, BLOCK_SIZE_M=BLOCK_SIZE_K,
                                                               BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCKED_OUTPUT=BLOCKED_B,
                                                               TRANSPOSE=TRANSPOSED_B, PACK32=PACKED_B)
        b = bb

    grid = ((M // BLOCK_SIZE_M) * (N // BLOCK_SIZE_N), )
    matmul_kernel_amx[grid](
        a, b, c,  #
        M, N, K,  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        GROUP_SIZE_M=GROUP_SIZE_M, GROUP_SIZE_N=GROUP_SIZE_N,  #
        BLOCKED_A=BLOCKED_A, BLOCKED_B=BLOCKED_B,  #
        TRANSPOSED_B=TRANSPOSED_B, PACKED_B=PACKED_B)

    torch.testing.assert_close(c, ref, atol=1e-2, rtol=0)
