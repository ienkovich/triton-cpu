"""
Matrix Multiplication
=====================
In this tutorial, you will write a very short high-performance FP32 matrix multiplication kernel.

You will specifically learn about:

* Block-level matrix multiplications.

* Multi-dimensional pointer arithmetic.

* Program re-ordering for improved L2 cache hit rate.

* Automatic performance tuning.

"""

# %%
# Motivations
# -----------
#
# Matrix multiplications are a key building block of most modern high-performance computing systems.
# They are notoriously hard to optimize, hence their implementation is generally done by
# hardware vendors themselves as part of so-called "kernel libraries" (e.g., cuBLAS).
# Unfortunately, these libraries are often proprietary and cannot be easily customized
# to accommodate the needs of modern deep learning workloads (e.g., fused activation functions).
# In this tutorial, you will learn how to implement efficient matrix multiplications by
# yourself with Triton, in a way that is easy to customize and extend.
#
# Roughly speaking, the kernel that we will write will implement the following blocked
# algorithm to multiply a (M, K) by a (K, N) matrix:
#
#  .. code-block:: python
#
#    # Do in parallel
#    for m in range(0, M, BLOCK_SIZE_M):
#      # Do in parallel
#      for n in range(0, N, BLOCK_SIZE_N):
#        acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
#        for k in range(0, K, BLOCK_SIZE_K):
#          a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
#          b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
#          acc += dot(a, b)
#        C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
#
# where each iteration of the doubly-nested for-loop is performed by a dedicated Triton program instance.

# %%
# Compute Kernel
# --------------
#
# The above algorithm is, actually, fairly straightforward to implement in Triton.
# The main difficulty comes from the computation of the memory locations at which blocks
# of :code:`A` and :code:`B` must be read in the inner loop. For that, we need
# multi-dimensional pointer arithmetic.
#
# Pointer Arithmetic
# ~~~~~~~~~~~~~~~~~~~
#
# For a row-major 2D tensor :code:`X`, the memory location of :code:`X[i, j]` is given
# by :code:`&X[i, j] = X + i*stride_xi + j*stride_xj`.
# Therefore, blocks of pointers for :code:`A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K]` and
# :code:`B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]` can be defined in pseudo-code as:
#
#  .. code-block:: python
#
#    &A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)[:, None]*A.stride(0) + (k : k+BLOCK_SIZE_K)[None, :]*A.stride(1);
#    &B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] =  b_ptr + (k : k+BLOCK_SIZE_K)[:, None]*B.stride(0) + (n : n+BLOCK_SIZE_N)[None, :]*B.stride(1);
#
# Which means that pointers for blocks of A and B can be initialized (i.e., :code:`k=0`) in Triton as the following
# code. Also note that we need an extra modulo to handle the case where :code:`M` is not a multiple of
# :code:`BLOCK_SIZE_M` or :code:`N` is not a multiple of :code:`BLOCK_SIZE_N`, in which case we can pad the data with
# some useless values, which will not contribute to the results. For the :code:`K` dimension, we will handle that later
# using masking load semantics.
#
#  .. code-block:: python
#
#    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#    offs_k = tl.arange(0, BLOCK_SIZE_K)
#    a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_k [None, :]*stride_ak)
#    b_ptrs = b_ptr + (offs_k [:, None]*stride_bk + offs_bn[None, :]*stride_bn)
#
# And then updated in the inner loop as follows:
#
#  .. code-block:: python
#
#    a_ptrs += BLOCK_SIZE_K * stride_ak;
#    b_ptrs += BLOCK_SIZE_K * stride_bk;
#
#
# L2 Cache Optimizations
# ~~~~~~~~~~~~~~~~~~~~~~
#
# As mentioned above, each program instance computes a :code:`[BLOCK_SIZE_M, BLOCK_SIZE_N]`
# block of :code:`C`.
# It is important to remember that the order in which these blocks are computed does
# matter, since it affects the L2 cache hit rate of our program, and unfortunately, a
# simple row-major ordering
#
#  .. code-block:: Python
#
#    pid = triton.program_id(0);
#    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M;
#    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N;
#    pid_m = pid / grid_n;
#    pid_n = pid % grid_n;
#
# is just not going to cut it.
#
# One possible solution is to launch blocks in an order that promotes data reuse.
# This can be done by 'super-grouping' blocks in groups of :code:`GROUP_M` rows before
# switching to the next column:
#
#  .. code-block:: python
#
#    # Program ID
#    pid = tl.program_id(axis=0)
#    # Number of program ids along the M axis
#    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#    # Number of programs ids along the N axis
#    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#    # Number of programs in group
#    num_pid_in_group = GROUP_SIZE_M * num_pid_n
#    # Id of the group this program is in
#    group_id = pid // num_pid_in_group
#    # Row-id of the first program in the group
#    first_pid_m = group_id * GROUP_SIZE_M
#    # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
#    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#    # *Within groups*, programs are ordered in a column-major order
#    # Row-id of the program in the *launch grid*
#    pid_m = first_pid_m + (pid % group_size_m)
#    # Col-id of the program in the *launch grid*
#    pid_n = (pid % num_pid_in_group) // group_size_m
#
# For example, in the following matmul where each matrix is 9 blocks by 9 blocks,
# we can see that if we compute the output in row-major ordering, we need to load 90
# blocks into SRAM to compute the first 9 output blocks, but if we do it in grouped
# ordering, we only need to load 54 blocks.
#
#   .. image:: grouped_vs_row_major_ordering.png
#
# In practice, this can improve the performance of our matrix multiplication kernel by
# more than 10\% on some hardware architecture (e.g., 220 to 245 TFLOPS on A100).
#

# %%
# Final Result
# ------------

import torch

import triton
import triton.language as tl

BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 32
GROUP_SIZE_M = 4
GROUP_SIZE_N = 4
USE_GPU = False


@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details

    #offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    #offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    #offs_k = tl.arange(0, BLOCK_SIZE_K)
    #a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    #b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    offs_m = pid_m * BLOCK_SIZE_M
    offs_n = pid_n * BLOCK_SIZE_N
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(K, 1), offsets=(offs_m, 0),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(N, 1), offsets=(0, offs_n),
                                    block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0))
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(N, 1), offsets=(offs_m, offs_n),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to matrix C's type after the loop, if C has lower precision type (for example, float16 and bfloat16).
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.

        # TODO: Currently masked load is not supported yet.
        # a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        # b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
        # Advance the ptrs to the next K block.
        #a_ptrs += BLOCK_SIZE_K * stride_ak
        #b_ptrs += BLOCK_SIZE_K * stride_bk
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    # Convert the accumulator to the output matrix C's type if needed.
    c = accumulator

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    #offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    #offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    #c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    # TODO: Currently masked load is not supported yet.
    # c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    # tl.store(c_ptrs, c, mask=c_mask)
    #tl.store(c_ptrs, c)
    tl.store(c_block_ptr, c)


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    #TODO: Currently masked load is not supported yet.
    assert (M % BLOCK_SIZE_M == 0) and (N % BLOCK_SIZE_N == 0) and (
        K % BLOCK_SIZE_K == 0), "Masking currently not supported, Matrix dimensions must be multiples of block size"
    if c is None:
        # Allocates output.
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    else:
        assert c.shape == (M, N), "Incompatible dimensions"
    # 1D launch kernel where each block gets its own program.
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        GROUP_SIZE_M=GROUP_SIZE_M,  #
    )
    return c

@triton.jit
def matmul_kernel_amx_interleaved(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        # number of blocks in a group
        GROUP_SIZE_M: tl.constexpr, GROUP_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    group_id = pid // (GROUP_SIZE_M * GROUP_SIZE_N)
    groups_n = N // BLOCK_SIZE_N // GROUP_SIZE_N
    group_m = group_id // groups_n
    group_n = group_id % groups_n
    block_id = pid % (GROUP_SIZE_M * GROUP_SIZE_N)
    block_m = block_id // GROUP_SIZE_N
    block_n = block_id % GROUP_SIZE_N

    offs_m = (group_m * GROUP_SIZE_M + block_m) * BLOCK_SIZE_M
    offs_n = (group_n * GROUP_SIZE_N + block_n) * BLOCK_SIZE_N

    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(K, 1), offsets=(offs_m, 0),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K // 2, N * 2), strides=(N * 2, 1), offsets=(0, offs_n * 2),
                                    block_shape=(BLOCK_SIZE_K // 2, BLOCK_SIZE_N * 2), order=(1, 0))
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(N, 1), offsets=(offs_m, offs_n),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)

        c += tl.dot(a, b, out_dtype=tl.float32, rhs_encoding="row_major_interleaved")

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K // 2, 0))

    tl.store(c_block_ptr, c)


@triton.jit
def interleave_kernel(in_p, out_p, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    tl.static_assert(N % BLOCK_SIZE == 0)
    for i in tl.range(0, tl.cdiv(M, 2)):
        for j in tl.range(0, tl.cdiv(N, BLOCK_SIZE)):
            row1 = tl.load(in_p + N * i * 2 + j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
            row2 = tl.load(in_p + N * (i * 2 + 1) + j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
            joined = tl.reshape(tl.join(row1, row2), (2 * BLOCK_SIZE, ))
            tl.store(out_p + N * i * 2 + j * BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE * 2), joined)


def matmul_amx_interleaved(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    # Check constraints.
    assert a.shape[1] == b.shape[0] * 2, "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    N = b.shape[1] // 2
    #TODO: Currently masked load is not supported yet.
    assert (M % (BLOCK_SIZE_M * GROUP_SIZE_M) == 0) and (N % (BLOCK_SIZE_N * GROUP_SIZE_N) == 0) and (
        K % BLOCK_SIZE_K == 0), "Masking currently not supported, Matrix dimensions must be multiples of block size"
    if c is None:
        # Allocates output.
        c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    else:
        assert c.shape == (M, N), "Incompatible dimensions"
    # 1D launch kernel where each block gets its own program.
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    matmul_kernel_amx_interleaved[grid](
        a, b, c,  #
        M, N, K,  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        GROUP_SIZE_M=GROUP_SIZE_M, GROUP_SIZE_N=GROUP_SIZE_N,#
    )
    return c


@triton.jit
def matmul_kernel_amx_interleaved_blocked(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        # number of blocks in a group
        GROUP_SIZE_M: tl.constexpr, GROUP_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    group_id = pid // (GROUP_SIZE_M * GROUP_SIZE_N)
    groups_n = N // BLOCK_SIZE_N // GROUP_SIZE_N
    group_m = group_id // groups_n
    group_n = group_id % groups_n
    block_id = pid % (GROUP_SIZE_M * GROUP_SIZE_N)
    block_m = group_m * GROUP_SIZE_M + block_id // GROUP_SIZE_N
    block_n = group_n * GROUP_SIZE_N + block_id % GROUP_SIZE_N

    offs_m = block_m * BLOCK_SIZE_M
    offs_n = block_n * BLOCK_SIZE_N

    #a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(K, 1), offsets=(offs_m, 0),
    #                                block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0))
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M * K // BLOCK_SIZE_K, BLOCK_SIZE_K), strides=(BLOCK_SIZE_K, 1),
                                    offsets=(block_m * BLOCK_SIZE_M * (K // BLOCK_SIZE_K), 0),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K * N // BLOCK_SIZE_N // 2, BLOCK_SIZE_N * 2),
                                    strides=(BLOCK_SIZE_N * 2, 1),
                                    # B blocks are in transposed order. To move by one block in N dimension
                                    # we need to skip BLOCK_SIZE_N former columns which are now K / 2 of
                                    # rows.
                                    offsets=(block_n * K // 2, 0),
                                    block_shape=(BLOCK_SIZE_K // 2, BLOCK_SIZE_N * 2), order=(1, 0))
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(N, 1), offsets=(offs_m, offs_n),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)

        c += tl.dot(a, b, out_dtype=tl.float32, rhs_encoding="row_major_interleaved")

        #a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        a_block_ptr = tl.advance(a_block_ptr, (BLOCK_SIZE_M, 0))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K // 2, 0))

    tl.store(c_block_ptr, c)

@triton.jit
def interleave_blocked_kernel(in_p, out_p, M: tl.constexpr, N: tl.constexpr,
                              BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    tl.static_assert(M % BLOCK_SIZE_M == 0)
    tl.static_assert(N % BLOCK_SIZE_N == 0)
    BLOCKS_N = N // BLOCK_SIZE_N
    m = tl.program_id(0) // BLOCKS_N
    n = tl.program_id(0) % BLOCKS_N
    BLOCK_IN_OFFS = m * BLOCK_SIZE_M * N + n * BLOCK_SIZE_N
    BLOCK_OUT_OFFS = m * BLOCK_SIZE_M * BLOCK_SIZE_N + n * BLOCK_SIZE_N * M
    for i in tl.range(0, BLOCK_SIZE_M // 2):
        row1 = tl.load(in_p + BLOCK_IN_OFFS + N * i * 2 + tl.arange(0, BLOCK_SIZE_N))
        row2 = tl.load(in_p + BLOCK_IN_OFFS + N * (i * 2 + 1) + tl.arange(0, BLOCK_SIZE_N))
        joined = tl.reshape(tl.join(row1, row2), (2 * BLOCK_SIZE_N,))
        tl.store(out_p + BLOCK_OUT_OFFS + i * BLOCK_SIZE_N * 2 + tl.arange(0, BLOCK_SIZE_N * 2), joined)


@triton.jit
def blocked_kernel(in_p, out_p, M: tl.constexpr, N: tl.constexpr,
                   BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    tl.static_assert(M % BLOCK_SIZE_M == 0)
    tl.static_assert(N % BLOCK_SIZE_N == 0)
    BLOCKS_N = N // BLOCK_SIZE_N
    m = tl.program_id(0) // BLOCKS_N
    n = tl.program_id(0) % BLOCKS_N
    BLOCK_IN_OFFS = m * BLOCK_SIZE_M * N + n * BLOCK_SIZE_N
    BLOCK_OUT_OFFS = tl.program_id(0) * BLOCK_SIZE_M * BLOCK_SIZE_N
    for i in tl.range(0, BLOCK_SIZE_M):
        row = tl.load(in_p + BLOCK_IN_OFFS + N * i + tl.arange(0, BLOCK_SIZE_N))
        tl.store(out_p + BLOCK_OUT_OFFS + BLOCK_SIZE_N * i + tl.arange(0, BLOCK_SIZE_N), row)


def matmul_amx_interleaved_blocked(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    # Check constraints.
    assert a.shape[1] == b.shape[0] * 2, "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    N = b.shape[1] // 2
    #print((M, N, K))
    #TODO: Currently masked load is not supported yet.
    assert (M % (BLOCK_SIZE_M * GROUP_SIZE_M) == 0) and (N % (BLOCK_SIZE_N * GROUP_SIZE_N) == 0) and (
        K % BLOCK_SIZE_K == 0), "Masking currently not supported, Matrix dimensions must be multiples of block size"
    if c is None:
        # Allocates output.
        c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    else:
        assert c.shape == (M, N), "Incompatible dimensions"
    # 1D launch kernel where each block gets its own program.
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    matmul_kernel_amx_interleaved_blocked[grid](
        a, b, c,  #
        M, N, K,  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        GROUP_SIZE_M=GROUP_SIZE_M, GROUP_SIZE_N=GROUP_SIZE_N,#
    )
    return c


@triton.jit
def prepack_kernel(in_p, out_p, M: tl.constexpr, N: tl.constexpr,
                   BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                   BLOCKED_OUTPUT: tl.constexpr, TRANSPOSE: tl.constexpr,
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
def matmul_kernel_amx(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        # number of blocks in a group
        GROUP_SIZE_M: tl.constexpr, GROUP_SIZE_N: tl.constexpr,
        BLOCKED_A: tl.constexpr, BLOCKED_B: tl.constexpr,
        TRANSPOSED_B: tl.constexpr, PACKED_B: tl.constexpr
):
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

    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M // BLOCK_SIZE_M, K // BLOCK_SIZE_K, BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    strides=(a_stride_block_m, a_stride_block_k, a_stride_m, a_stride_k), offsets=(block_m, 0, 0, 0),
                                    block_shape=(1, 1, BLOCK_SIZE_M, BLOCK_SIZE_K), order=(3, 2, 1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K // BLOCK_SIZE_K, N // BLOCK_SIZE_N, PACKED_BLOCK_SIZE_K, PACKED_BLOCK_SIZE_N),
                                    strides=(b_stride_block_k, b_stride_block_n, b_stride_k, b_stride_n), offsets=(0, block_n, 0, 0),
                                    block_shape=(1, 1, PACKED_BLOCK_SIZE_K, PACKED_BLOCK_SIZE_N), order=(3, 2, 1, 0))
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(N, 1), offsets=(block_m * BLOCK_SIZE_M, block_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_block_ptr).reshape((BLOCK_SIZE_M, BLOCK_SIZE_K))
        b = tl.load(b_block_ptr).reshape((PACKED_BLOCK_SIZE_K, PACKED_BLOCK_SIZE_N))

        c += tl.dot(a, b, out_dtype=tl.float32, rhs_encoding="row_major_interleaved" if PACKED_B else "row_major")

        a_block_ptr = tl.advance(a_block_ptr, (0, 1, 0, 0))
        b_block_ptr = tl.advance(b_block_ptr, (1, 0, 0, 0))

    tl.store(c_block_ptr, c)


def matmul_amx(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, BLOCKED_A, BLOCKED_B, TRANSPOSED_B, PACKED_B):
    M, K = a.shape
    N = c.shape[1]
    #TODO: Currently masked load is not supported yet.
    assert (M % (BLOCK_SIZE_M * GROUP_SIZE_M) == 0) and (N % (BLOCK_SIZE_N * GROUP_SIZE_N) == 0) and (
        K % BLOCK_SIZE_K == 0), "Masking currently not supported, Matrix dimensions must be multiples of block size"
    if c is None:
        # Allocates output.
        c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    else:
        assert c.shape == (M, N), "Incompatible dimensions"
    # 1D launch kernel where each block gets its own program.
    grid = ((M // BLOCK_SIZE_M) * (N // BLOCK_SIZE_N),)
    matmul_kernel_amx[grid](
        a, b, c,  #
        M, N, K,  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        GROUP_SIZE_M=GROUP_SIZE_M, GROUP_SIZE_N=GROUP_SIZE_N,  #
        BLOCKED_A=BLOCKED_A, BLOCKED_B=BLOCKED_B,  #
        TRANSPOSED_B=TRANSPOSED_B, PACKED_B=PACKED_B
    )
    return c


# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation.

torch.manual_seed(0)

triton.runtime.driver.set_active_to_cpu()

a = torch.randn((512, 512), device='cpu', dtype=torch.bfloat16)
b = torch.randn((512, 512), device='cpu', dtype=torch.bfloat16)
triton_output = matmul(a, b, None)
torch_output = torch.matmul(a.to(torch.float32), b.to(torch.float32))
rtol = 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ TritonCPU and TorchCPU match")
else:
    print("❌ TritonCPU and TorchCPU differ, the maximum difference is "
          f'{torch.max(torch.abs(triton_output - torch_output))}')
bi = torch.empty((256, 1024), dtype=torch.bfloat16, device='cpu')
interleave_kernel[(1,)](b, bi, 512, 512, BLOCK_SIZE=64)
triton_output = matmul_amx_interleaved(a, bi, None)
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ TritonCPU AMX and TorchCPU match")
else:
    print("❌ TritonCPU AMX and TorchCPU differ, the maximum difference is "
          f'{torch.max(torch.abs(triton_output - torch_output))}')
bib = torch.empty((256, 1024), dtype=torch.bfloat16, device='cpu')
interleave_blocked_kernel[((512 // BLOCK_SIZE_K) * (512 // BLOCK_SIZE_N),)](b, bib, 512, 512, BLOCK_SIZE_M=BLOCK_SIZE_K, BLOCK_SIZE_N=BLOCK_SIZE_N)
ab = torch.empty((512, 512), dtype=torch.bfloat16, device='cpu')
blocked_kernel[((512 // BLOCK_SIZE_M) * (512 // BLOCK_SIZE_K),)](a, ab, 512, 512, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_K)
triton_output = matmul_amx_interleaved_blocked(ab, bib, None)
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ TritonCPU AMX blocked and TorchCPU match")
else:
    print("❌ TritonCPU AMX blocked and TorchCPU differ, the maximum difference is "
          f'{torch.max(torch.abs(triton_output - torch_output))}')

# %%
# Benchmark
# ---------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now compare the performance of our kernel against that of Pytorch. Here we focus on square matrices,
# but feel free to arrange this script as you wish to benchmark any other matrix shape.

LINE_VALS = ['triton-cpu-single', 'triton-cpu', 'triton-cpu-prepack-single', 'triton-cpu-prepack', 'triton-cpu-prepack-blocked-single', 'triton-cpu-prepack-blocked', 'torch-cpu-native', 'torch-cpu-compile']
#LINE_VALS = ['triton-cpu-single', 'triton-cpu']
LINE_NAMES = ['TritonCPU 1', 'TritonCPU', 'TritonCPU Prepack 1', 'TritonCPU Prepack', 'TritonCPU Prepack Blocked 1', 'TritonCPU Prepack Blocked', 'TorchCPU (native)', 'TorchCPU (compile)']
LINE_STYLES = [('blue', '--'), ('blue', '-'), ('blue', '--'), ('blue', '-'), ('blue', '--'), ('blue', '-'), ('green', '--'), ('green', '-')]

AMX_OPTS = ['nopack', 'pack32', 'blockeda', 'blockedb', 'blockeda-blockedb', 'blockeda-blockedb-transposedb', 'blockeda-blockedb-pack32', 'blockeda-blockedb-transposedb-pack32']
LINE_VALS = [f'triton-cpu-{opt}{prefix}' for prefix in ['-single', ''] for opt in AMX_OPTS] + ['torch-cpu-native']
LINE_NAMES = LINE_VALS
LINE_STYLES = [('blue', '--')] * len(LINE_VALS)

if USE_GPU and triton.runtime.driver.get_active_gpus():
    triton.runtime.driver.set_active_to_gpu()
    a = a.to('cuda')
    b = b.to('cuda')
    triton_output = matmul(a, b, None)
    torch_output = torch.matmul(a, b)
    print(f"triton_gpu_output_with_{a.dtype}_inputs={triton_output}")
    print(f"torch_gpu_output_with_{a.dtype}_inputs={torch_output}")
    rtol = 0
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ TritonGPU and TorchGPU match")
    else:
        print("❌ TritonGPU and TorchGPU differ, the maximum difference is "
              f'{torch.max(torch.abs(triton_output - torch_output))}')

    LINE_VALS += ['triton-gpu', 'torch-gpu']
    LINE_NAMES += ['TritonGPU', 'TorchGPU']
    LINE_STYLES += [('yellow', '-'), ('red', '-')]

# %%
# Seems like we're good to go!

# %%
# Benchmark
# ---------
#
# We can now benchmark our custom op on vectors of increasing sizes to get a sense of how it does relative to PyTorch.
# To make things easier, Triton has a set of built-in utilities that allow us to concisely plot the performance of our custom ops.
# for different problem sizes.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 21)],  # Different possible values for `x_name`
        #x_vals=[128 * i for i in range(2, 21, 4)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=LINE_VALS,  # Possible values for `line_arg`.
        line_names=LINE_NAMES,  # Label name for the lines.
        styles=LINE_STYLES,  # Line styles.
        ylabel='GFLOPS',  # Label name for the y-axis.
        plot_name=
        # Name for the plot. Used also as a file name for saving the plot.
        f'matmul-performance-bf16 (BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, BLOCK_SIZE_K={BLOCK_SIZE_K}, GROUP_SIZE_M={GROUP_SIZE_M})',
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(M, N, K, provider):
    import os

    device = 'cpu' if 'cpu' in provider else 'cuda'
    a = torch.randn((M, K), device=device, dtype=torch.bfloat16)
    b = torch.randn((K, N), device=device, dtype=torch.bfloat16)

    if device == 'cpu':
        if 'triton' in provider:
            c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
            if 'blockeda' in provider:
                ab = torch.empty_like(a)
                prepack_kernel[(M // BLOCK_SIZE_M, K // BLOCK_SIZE_K)](a, ab, M, K, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_K, BLOCKED_OUTPUT=True, TRANSPOSE=False, PACK32=False)
                a = ab
            if 'blockedb' in provider or 'pack32' in provider:
                bb = torch.empty_like(b)
                prepack_kernel[(K // BLOCK_SIZE_K, N // BLOCK_SIZE_N)](b, bb, K, N, BLOCK_SIZE_M=BLOCK_SIZE_K, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCKED_OUTPUT='blockedb' in provider, TRANSPOSE='transposedb' in provider, PACK32='pack32' in provider)
                b = bib
        else:
            c = torch.zeros((M, N), device=a.device, dtype=torch.bfloat16)
        triton.runtime.driver.set_active_to_cpu()
        if 'single' in provider:
            os.environ['TRITON_CPU_SINGLE_CORE'] = '1'
        else:
            os.unsetenv('TRITON_CPU_SINGLE_CORE')
    else:
        c = None
        triton.runtime.driver.set_active_to_gpu()

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-gpu':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles,
                                                     device_type=device)
    elif provider == 'triton-gpu':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, None), quantiles=quantiles,
                                                     device_type=device)
    elif provider == 'torch-cpu-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b, out=c), quantiles=quantiles,
                                                     device_type=device)
    elif provider == 'torch-cpu-compile':
        compiled = torch.compile(torch.matmul)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: compiled(a, b, out=c), quantiles=quantiles,
                                                     device_type=device)
    elif provider == 'triton-cpu-single':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, c), quantiles=quantiles, device_type=device)
    elif 'triton-cpu' in provider:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_amx(a, b, c, 'blockeda' in provider, 'blockedb' in provider, 'transposedb' in provider, 'pack32' in provider), quantiles=quantiles, device_type=device)
    elif provider == 'triton-cpu':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, c), quantiles=quantiles, device_type=device)
    elif provider == 'triton-cpu-prepack-single':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_amx_interleaved(a, b, c), quantiles=quantiles, device_type=device)
    elif provider == 'triton-cpu-prepack':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_amx_interleaved(a, b, c), quantiles=quantiles, device_type=device)
    elif provider == 'triton-cpu-prepack-blocked-single':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_amx_interleaved_blocked(a, b, c), quantiles=quantiles, device_type=device)
    elif provider == 'triton-cpu-prepack-blocked':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_amx_interleaved_blocked(a, b, c), quantiles=quantiles, device_type=device)
    perf = lambda ms: 2 * M * N * K * 1e-9 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=True)
