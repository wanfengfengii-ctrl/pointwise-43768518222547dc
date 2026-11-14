"""Denoising demo and scaling benchmark for tv.denoise.

  part 1: step signal + noise -> TV denoise (uniform weights/penalties)
  part 2: same with per-point reliability weights and smoothing strengths
  part 3: doubling benchmark up to n = 200000: wall time, peak auxiliary
          memory (tracemalloc), and KKT residuals at full scale

Run:  python3 demo_tv.py
"""

import random
import time
import tracemalloc

import tv
from verify_tv import kkt_residuals


def rmse(a, b):
    return (sum((u - v) ** 2 for u, v in zip(a, b)) / len(a)) ** 0.5


def ascii_plot(cols_true, cols_noisy_lo, cols_noisy_hi, cols_den,
               lo, hi, width, height):
    """Render column aggregates; noisy shown as its [min,max] band."""
    span = hi - lo or 1.0

    def row_of(v):
        return min(height - 1, max(0, int((hi - v) / span * (height - 1)
                                          + 0.5)))
    grid = [[" "] * width for _ in range(height)]
    for c in range(width):
        r1, r2 = row_of(cols_noisy_lo[c]), row_of(cols_noisy_hi[c])
        for r in range(min(r1, r2), max(r1, r2) + 1):
            grid[r][c] = "."
    for c in range(width):
        grid[row_of(cols_true[c])][c] = "="
    for c in range(width):
        grid[row_of(cols_den[c])][c] = "#"
    print("    +" + "-" * width + "+")
    for r in range(height):
        v = hi - span * r / (height - 1)
        print("%+5.1f |%s|" % (v, "".join(grid[r])))
    print("    +" + "-" * width + "+")
    print("      legend: '=' true step, '#' denoised, '.' noisy min/max band")


def demo_uniform():
    rng = random.Random(20261114)
    n = 2000
    jumps = (500, 1000, 1500)
    levels = (0.0, 3.0, -2.0, 1.5)
    true = []
    for i in range(n):
        j = sum(i >= g for g in jumps)
        true.append(levels[j])
    noisy = [v + rng.gauss(0.0, 0.8) for v in true]
    w = [1.0] * n
    lam = [2.0] * (n - 1)
    den = tv.denoise(noisy, w, lam)

    print("part 1: step signal, n=%d, gaussian noise sigma=0.8, "
          "uniform w=1, lambda=2" % n)
    print("  RMSE noisy    vs true: %.4f" % rmse(noisy, true))
    print("  RMSE denoised vs true: %.4f" % rmse(den, true))
    for g in jumps:
        err = max(abs(den[g + d] - true[g + d]) for d in (-2, -1, 0, 1, 2))
        print("  jump @%4d: true step %+.1f -> %+.1f, denoised step %+.3f "
              "-> %+.3f, local max err %.3g"
              % (g, true[g - 1], true[g], den[g - 1], den[g], err))

    width, height = 96, 16
    seg = n // width
    cols_true, cols_lo, cols_hi, cols_den = [], [], [], []
    for c in range(width):
        chunk_n = noisy[c * seg:(c + 1) * seg]
        cols_true.append(sum(true[c * seg:(c + 1) * seg]) / seg)
        cols_lo.append(min(chunk_n))
        cols_hi.append(max(chunk_n))
        cols_den.append(sum(den[c * seg:(c + 1) * seg]) / seg)
    ascii_plot(cols_true, cols_lo, cols_hi, cols_den,
               min(cols_lo), max(cols_hi), width, height)


def demo_weighted():
    rng = random.Random(998877)
    n = 1500
    true = [0.0] * 500 + [2.5] * 500 + [-1.5] * 500
    # unreliable sensor stretches get small weights; trusted ones large
    w = [0.3 if (i // 100) % 3 == 1 else 2.0 for i in range(n)]
    noisy = [t + rng.gauss(0.0, 1.0 if wi < 1 else 0.4)
             for t, wi in zip(true, w)]
    # stronger smoothing where the signal is flat-ish, weaker near jumps
    lam = [3.0] * (n - 1)
    for g in (500, 1000):
        for i in range(max(0, g - 3), min(n - 1, g + 3)):
            lam[i] = 0.2
    den = tv.denoise(noisy, w, lam)
    print("part 2: per-point weights (0.3 unreliable / 2.0 trusted) and "
          "locally relaxed penalties near jumps")
    print("  RMSE noisy    vs true: %.4f" % rmse(noisy, true))
    print("  RMSE denoised vs true: %.4f" % rmse(den, true))
    for g in (500, 1000):
        print("  jump @%4d: denoised step %+.3f -> %+.3f (true %+.1f -> "
              "%+.1f)" % (g, den[g - 1], den[g], true[g - 1], true[g]))


def benchmark():
    print("part 3: doubling benchmark (random data, fixed seed)")
    print("  %8s %10s %8s %12s %14s" % ("n", "time [s]", "x/prev",
                                        "peak mem [MB]", "KKT max res."))
    rng = random.Random(424242)
    prev_t = None
    for n in (12500, 25000, 50000, 100000, 200000):
        y = [rng.uniform(-10000, 10000) for _ in range(n)]
        w = [rng.uniform(0.1, 10) for _ in range(n)]
        lam = [rng.uniform(0, 100) for _ in range(n - 1)]
        best = None
        for _ in range(3):
            t0 = time.perf_counter()
            tv.denoise(y, w, lam)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        tracemalloc.start()
        x = tv.denoise(y, w, lam)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        res = max(kkt_residuals(y, w, lam, x))
        ratio = "" if prev_t is None else "%.2f" % (best / prev_t)
        print("  %8d %10.4f %8s %12.1f %14.3g"
              % (n, best, ratio, peak / 1e6, res))
        prev_t = best
    print("  (time ~doubles and peak memory ~doubles per size doubling:"
          " linear time and linear auxiliary space)")


def main():
    demo_uniform()
    print()
    demo_weighted()
    print()
    benchmark()


if __name__ == "__main__":
    main()
