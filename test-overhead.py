import time
import torch
import triton
import triton.language as tl

@triton.jit
def test1(x):
    tl.store(x, float(1))

@triton.jit
def test2(x, t1):
    tl.store(x, float(1))

@triton.jit
def test3(x, t1, t2):
    tl.store(x, float(1))

@triton.jit
def test4(x, t1, t2, t3):
    tl.store(x, float(1))

@triton.jit
def test5(x, t1, t2, t3, t4):
    tl.store(x, float(1))

@triton.jit
def test6(x, t1, t2, t3, t4, t5):
    tl.store(x, float(1))

@triton.jit
def test7(x, t1, t2, t3, t4, t5, t6):
    tl.store(x, float(1))

@triton.jit
def test8(x, t1, t2, t3, t4, t5, t6, t7):
    tl.store(x, float(1))

@triton.jit
def test9(x, t1, t2, t3, t4, t5, t6, t7, t8):
    tl.store(x, float(1))

@triton.jit
def test10(x, t1, t2, t3, t4, t5, t6, t7, t8, t9):
    tl.store(x, float(1))

@triton.jit
def layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([16], dtype=tl.float32)
    for off in range(0, N, 16):
        cols = off + tl.arange(0, 16)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([16], dtype=tl.float32)
    for off in range(0, N, 16):
        cols = off + tl.arange(0, 16)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, 16):
        cols = off + tl.arange(0, 16)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)

def get_median(vals):
    return torch.quantile(torch.tensor(vals, dtype=torch.float), torch.tensor([0.0, 0.5, 1.0], dtype=torch.float)).tolist()

def measure_overhead(kernel, num_args: int, rep=100):
    x = torch.zeros((1, ), dtype=torch.float32)
    additional_args = tuple(torch.zeros((4096 * 4096, ), dtype=torch.float32) for i in range(0, num_args - 1))
    if num_args == 9:
        additional_args = (additional_args[0], additional_args[1], additional_args[2], additional_args[3], additional_args[4], 100, 100, 0.1)

    # Warmup
    kernel[(1,)](x, *additional_args)
    # Measure
    call_times = []
    launcher_times = []
    overhead = []
    time0 = []
    time1 = []
    time2 = []
    time3 = []
    time4 = []
    time5 = []
    time6 = []
    time7 = []
    kernel_call = []
    for _ in range(0, rep):
        start = time.perf_counter_ns()
        meta, t0, t1, t2, t3, t4, t5, t6, t7 = kernel[(1,)](x, *additional_args)
        end = time.perf_counter_ns()
        #print(f"Call time: {(end-start) / 1000}\nLauncher time: {x.item()}")
        call_times.append((end-start) / 1000)
        launcher_times.append(x.item())
        overhead.append((end-start) / 1000 - x.item())
        time0.append(t0 / 1000)
        time1.append(t1 / 1000)
        time2.append(t2 / 1000)
        time3.append(t3 / 1000)
        time4.append(t4 / 1000)
        time5.append(t5 / 1000)
        time6.append(t6 / 1000)
        time7.append(t7 / 1000)
        kernel_call.append((t7 - t6) / 1000)

    med_overhead = get_median(overhead)
    med_call_time = get_median(call_times)
    med_launcher_time = get_median(launcher_times)
    print(f"Call with {num_args} args\n  Median call time: {med_call_time}\n  Median launcher time: {med_launcher_time}\n  Median overhead: {med_overhead}")
    print(f"  Time0: {get_median(time0)}")
    print(f"  Time1: {get_median(time1)}")
    print(f"  Time2: {get_median(time2)}")
    print(f"  Time3: {get_median(time3)}")
    print(f"  Time4: {get_median(time4)}")
    print(f"  Time5: {get_median(time5)}")
    print(f"  Time6: {get_median(time6)}")
    print(f"  Time7: {get_median(time7)}")
    print(f"  Kenel call: {get_median(kernel_call)}")


measure_overhead(test1, 1)
measure_overhead(test2, 2)
measure_overhead(test3, 3)
measure_overhead(test4, 4)
measure_overhead(test5, 5)
measure_overhead(test6, 6)
measure_overhead(test7, 7)
measure_overhead(test8, 8)
measure_overhead(test9, 9)
measure_overhead(test10, 10)
measure_overhead(layer_norm_fwd_fused, 9)
