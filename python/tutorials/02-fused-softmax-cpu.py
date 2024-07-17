"""
Fused Softmax
=============

In this tutorial, you will write a fused softmax operation that is significantly faster
than PyTorch's native op for a particular class of matrices: those whose rows can fit in
the GPU's SRAM.

In doing so, you will learn about:

* The benefits of kernel fusion for bandwidth-bound operations.

* Reduction operators in Triton.

"""

# %%
# Motivations
# -----------
#
# Custom GPU kernels for elementwise additions are educationally valuable but won't get you very far in practice.
# Let us consider instead the case of a simple (numerically stabilized) softmax operation:
import os
import shutil

import torch

import triton
import triton.language as tl

USE_GPU = False


@torch.jit.script
def naive_softmax(x, y):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    #ret = numerator / denominator[:, None]
    torch.div(numerator, denominator[:, None], out=y)
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    #return ret

def softmax_for_compile(x, y):
    y = torch.softmax(x, axis=-1, out=y)
    return y

# %%
# When implemented naively in PyTorch, computing :code:`y = naive_softmax(x)` for :math:`x \in R^{M \times N}`
# requires reading :math:`5MN + 2M` elements from DRAM and writing back :math:`3MN + 2M` elements.
# This is obviously wasteful; we'd prefer to have a custom "fused" kernel that only reads
# X once and does all the necessary computations on-chip.
# Doing so would require reading and writing back only :math:`MN` bytes, so we could
# expect a theoretical speed-up of ~4x (i.e., :math:`(8MN + 4M) / 2MN`).
# The `torch.jit.script` flags aims to perform this kind of "kernel fusion" automatically
# but, as we will see later, it is still far from ideal.

# %%
# Compute Kernel
# --------------
#
# Our softmax kernel works as follows: each program loads a row of the input matrix X,
# normalizes it and writes back the result to the output Y.
#
# Note that one important limitation of Triton is that each block must have a
# power-of-two number of elements, so we need to internally "pad" each row and guard the
# memory operations properly if we want to handle any possible input shapes:

@triton.jit
def softmax_kernel_orig(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                        num_stages: tl.constexpr):
    # starting row of the program
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

@triton.jit
def softmax_kernel_pers(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                        num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

def softmax_orig(x, y=None):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software piepling stages.
    num_stages = 4 #if SIZE_SMEM > 200000 else 2

    # Allocate output
    if y is None:
        y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    #kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    #if kernel is None:
    #    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
    #                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    #    kernel._init_handles()
    #    n_regs = kernel.n_regs
    #    size_smem = kernel.metadata.shared
    #    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    #    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    #    num_programs = NUM_SM * occupancy
    #    kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(n_rows, 64)

    # Create a number of persistent programs.
    softmax_kernel_orig[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
    )
    return y

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, TILE_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    max_vec = tl.full((TILE_SIZE, ), -float('inf'), tl.float32)
    for i in range(0, tl.cdiv(n_cols + TILE_SIZE - 1, TILE_SIZE)):
        tile_offsets = tl.arange(0, TILE_SIZE) + i * TILE_SIZE
        input_ptrs = row_start_ptr + tile_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        tile = tl.load(input_ptrs, mask=tile_offsets < n_cols, other=-float('inf'))
        max_vec = max(max_vec, tile)
    max_val = tl.max(max_vec, axis=0)
    denominator = float(0)
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    accumulator = tl.zeros((TILE_SIZE, ), dtype=tl.float32)
    for i in range(0, tl.cdiv(n_cols + TILE_SIZE - 1, TILE_SIZE)):
        tile_offsets = tl.arange(0, TILE_SIZE) + i * TILE_SIZE
        input_ptrs = row_start_ptr + tile_offsets
        tile = tl.load(input_ptrs, mask=tile_offsets < n_cols, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = tile - max_val
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        accumulator = accumulator + numerator
        output_ptrs = output_row_start_ptr + tile_offsets
        tl.store(output_ptrs, numerator, mask=tile_offsets < n_cols)
    denominator = tl.sum(accumulator, axis=0)
    scale = 1 / denominator
    for i in range(0, tl.cdiv(n_cols + TILE_SIZE - 1, TILE_SIZE)):
        tile_offsets = tl.arange(0, TILE_SIZE) + i * TILE_SIZE
        #input_ptrs = row_start_ptr + tile_offsets
        output_ptrs = output_row_start_ptr + tile_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        tile = tl.load(output_ptrs, mask=tile_offsets < n_cols, other=-float('inf'))
        softmax_output = tile * scale
        # Write back output to DRAM
        tl.store(output_ptrs, softmax_output, mask=tile_offsets < n_cols)


@triton.jit
def softmax_kernel_pers_tiled(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, TILE_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    rows_per_prog = tl.cdiv(n_cols + tl.num_programs(0) - 1,  tl.num_programs(0))
    row_start = tl.program_id(0) * rows_per_prog
    row_end = min(row_start + rows_per_prog, n_rows)
    for row_idx in range(row_start, row_end):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        max_vec = tl.full((TILE_SIZE, ), -float('inf'), tl.float32)
        for i in range(0, tl.cdiv(n_cols + TILE_SIZE - 1, TILE_SIZE)):
            tile_offsets = tl.arange(0, TILE_SIZE) + i * TILE_SIZE
            input_ptrs = row_start_ptr + tile_offsets
            # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
            tile = tl.load(input_ptrs, mask=tile_offsets < n_cols, other=-float('inf'))
            max_vec = max(max_vec, tile)
        max_val = tl.max(max_vec, axis=0)
        denominator = float(0)
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        accumulator = tl.zeros((TILE_SIZE, ), dtype=tl.float32)
        for i in range(0, tl.cdiv(n_cols + TILE_SIZE - 1, TILE_SIZE)):
            tile_offsets = tl.arange(0, TILE_SIZE) + i * TILE_SIZE
            input_ptrs = row_start_ptr + tile_offsets
            tile = tl.load(input_ptrs, mask=tile_offsets < n_cols, other=-float('inf'))
            # Subtract maximum for numerical stability
            row_minus_max = tile - max_val
            # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
            numerator = tl.exp(row_minus_max)
            accumulator = accumulator + numerator
            output_ptrs = output_row_start_ptr + tile_offsets
            tl.store(output_ptrs, numerator, mask=tile_offsets < n_cols)
        denominator = tl.sum(accumulator, axis=0)
        scale = 1 / denominator
        for i in range(0, tl.cdiv(n_cols + TILE_SIZE - 1, TILE_SIZE)):
            tile_offsets = tl.arange(0, TILE_SIZE) + i * TILE_SIZE
            #input_ptrs = row_start_ptr + tile_offsets
            output_ptrs = output_row_start_ptr + tile_offsets
            # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
            tile = tl.load(output_ptrs, mask=tile_offsets < n_cols, other=-float('inf'))
            softmax_output = tile * scale
            # Write back output to DRAM
            tl.store(output_ptrs, softmax_output, mask=tile_offsets < n_cols)

# %%
# We can create a helper function that enqueues the kernel and its (meta-)arguments for any given input tensor.


def softmax(x, y=None):
    n_rows, n_cols = x.shape
    # The block size is the smallest power of two greater than the number of columns in `x`
    TILE_SIZE = 16
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 4
    #if BLOCK_SIZE >= 2048:
    #    num_warps = 8
    #if BLOCK_SIZE >= 4096:
    #    num_warps = 16
    # Allocate output
    if y is None:
        y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row of
    # the input matrix
    softmax_kernel[(n_rows, )](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        TILE_SIZE=TILE_SIZE,
    )
    return y


def softmax_pers_tiled(x, y=None):
    n_rows, n_cols = x.shape
    # The block size is the smallest power of two greater than the number of columns in `x`
    TILE_SIZE = 128
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 4
    #if BLOCK_SIZE >= 2048:
    #    num_warps = 8
    #if BLOCK_SIZE >= 4096:
    #    num_warps = 16
    # Allocate output
    if y is None:
        y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row of
    # the input matrix
    softmax_kernel_pers_tiled[(64, )](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        num_warps=num_warps,
        TILE_SIZE=TILE_SIZE,
    )
    return y

# %%
# Unit Test
# ---------

# %%
# We make sure that we test our kernel on a matrix with an irregular number of rows and columns.
# This will allow us to verify that our padding mechanism works.

triton.runtime.driver.set_active_to_cpu()

"""
torch.manual_seed(0)
x = torch.randn(1823, 781, device='cpu')
y_torch_cpu = torch.softmax(x, axis=1)
y_triton_cpu = softmax_orig(x)
assert torch.allclose(y_triton_cpu, y_torch_cpu), (y_triton_cpu, y_torch_cpu)
y_triton_cpu = softmax(x)
assert torch.allclose(y_triton_cpu, y_torch_cpu), (y_triton_cpu, y_torch_cpu)
"""

LINE_VALS = [
#    'triton-cpu-single',
#    'triton-cpu-orig-nomvec',
#    'triton-cpu-orig-mvec',
#    'triton-cpu-tiled-nomvec',
    'triton-cpu-tiled-mvec',
#    'triton-cpu-tiled-pers-mvec',
    'torch-cpu-compile',
#    'torch-cpu-jit',
    'torch-cpu-native',
]
#LINE_NAMES = ['Triton', 'Triton(mvec)', 'Triton(tiled)', 'Triton(tiled+mvec)', 'TorchInductor', 'TorchNative']
#LINE_STYLES = [('blue', '-'), ('blue', '-'), ('green', '-'), ('green', '--'), ('green', '--'), ('green', '-.')]
LINE_NAMES = LINE_VALS
LINE_STYLES = [('blue', '-')] * len(LINE_NAMES)

if USE_GPU and triton.runtime.driver.get_active_gpus():
    triton.runtime.driver.set_active_to_gpu()
    x = x.to('cuda')
    y_triton_gpu = softmax(x)
    y_torch_gpu = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton_gpu, y_torch_gpu), (y_triton_gpu, y_torch_gpu)
    LINE_VALS += ['triton-gpu', 'torch-gpu-native', 'torch-gpu-jit']
    LINE_NAMES += ['TritonGPU', 'TorchGPU (native)', 'TorchGPU (jit)']
    LINE_STYLES += [('yellow', '-'), ('red', '-'), ('red', '--')]

tmpdir = ".tmp"
def reset_cache_dir():
    os.environ["TRITON_CACHE_DIR"] = tmpdir
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)
    softmax_kernel.cache.clear()
    softmax_kernel_orig.cache.clear()
    softmax_kernel_pers.cache.clear()

# %%
# As expected, the results are identical.

# %%
# Benchmark
# ---------
#
# Here we will benchmark our operation as a function of the number of columns in the input matrix -- assuming 4096 rows.
# We will then compare its performance against (1) :code:`torch.softmax` and (2) the :code:`naive_softmax` defined above.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        #x_vals=[128 * i for i in range(2, 34, 1)],  # different possible values for `x_name`
        x_vals=[1024],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=LINE_VALS,  # Possible values for `line_arg`.
        line_names=LINE_NAMES,  # Label name for the lines.
        styles=LINE_STYLES,  # Line styles.
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    import os

    # Currently compilation time is very long. Let's show the progress.
    print(f"Running {provider} with {M} x {N}...")

    device = 'cpu' if 'cpu' in provider else 'cuda'
    x = torch.randn(M, N, device=device, dtype=torch.float32)

    if device == 'cpu':
        y = torch.empty_like(x)
        triton.runtime.driver.set_active_to_cpu()
        if 'single' in provider:
            os.environ['TRITON_CPU_SINGLE_CORE'] = '1'
        else:
            os.environ.pop('TRITON_CPU_SINGLE_CORE', None)

        if 'nomvec' in provider:
            os.environ['TRITON_CPU_NO_LIBMVEC'] = '1'
        else:
            os.environ.pop('TRITON_CPU_NO_LIBMVEC', None)

        #reset_cache_dir()
    else:
        y = None
        triton.runtime.driver.set_active_to_gpu()


    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-cpu-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1, out=y), quantiles=quantiles,
                                                     device_type=device)
    if provider == 'torch-cpu-jit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles, device_type=device)
    if provider == 'torch-cpu-compile':
        compiled = torch.compile(softmax_for_compile)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: compiled(x, y), quantiles=quantiles, device_type=device)
    if provider == 'triton-cpu-single':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x, y), quantiles=quantiles, device_type=device)
    if provider == 'triton-cpu-orig-nomvec':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax_orig(x, y), quantiles=quantiles, device_type=device)
    if provider == 'triton-cpu-orig-mvec':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax_orig(x, y), quantiles=quantiles, device_type=device)
    if provider == 'triton-cpu-tiled-nomvec':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x, y), quantiles=quantiles, device_type=device)
    if provider == 'triton-cpu-tiled-mvec':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x, y), quantiles=quantiles, device_type=device)
    if provider == 'triton-cpu-tiled-pers-mvec':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax_pers_tiled(x, y), quantiles=quantiles, device_type=device)
    if provider == 'triton-gpu':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles, device_type=device)
    if provider == 'torch-gpu-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles,
                                                     device_type=device)
    if provider == 'torch-gpu-jit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles, device_type=device)
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(show_plots=True, print_data=True)

# %%
# In the above plot, we can see that:
#  - Triton is 4x faster than the Torch JIT. This confirms our suspicions that the Torch JIT does not do any fusion here.
#  - Triton is noticeably faster than :code:`torch.softmax` -- in addition to being **easier to read, understand and maintain**.
#    Note however that the PyTorch `softmax` operation is more general and will work on tensors of any shape.
