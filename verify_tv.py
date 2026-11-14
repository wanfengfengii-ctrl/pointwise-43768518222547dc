"""Correctness verification for tv.denoise.

1. Supplied examples and structural edge cases.
2. KKT residual checks on random instances -- for this strongly convex
   problem the KKT conditions are necessary AND sufficient, so vanishing
   residuals certify the exact optimum.
3. Independent cross-check against a brute-force exact solver
   (enumeration of all jump sign patterns) for tiny n.
4. Input-validation and no-mutation checks.

Run:  python3 verify_tv.py
"""

import itertools
import random
import sys

import tv


# ---------------------------------------------------------------- helpers

def objective(y, w, lam, x):
    data = sum(wi * (xi - yi) ** 2 for wi, xi, yi in zip(w, x, y)) / 2.0
    pen = sum(li * abs(b - a) for li, a, b in zip(lam, x, x[1:]))
    return data + pen


def kkt_residuals(y, w, lam, x):
    """Max residuals of the KKT system (all should be ~0).

    z_i = sum_{k<=i} w_k (x_k - y_k)  is the dual variable of edge i.
      * stationarity:  z_{n-1} == 0
      * dual feas.:    |z_i| <= lam_i
      * compl. slack:  min(lam_i - |z_i|, |x_{i+1} - x_i|) == 0
      * sign match:    z_i * (x_{i+1} - x_i) >= 0
    """
    n = len(y)
    z = 0.0
    r_stat = r_dual = r_cslack = r_sign = 0.0
    for i in range(n):
        z += w[i] * (x[i] - y[i])
        if i == n - 1:
            r_stat = abs(z)
        else:
            li = lam[i]
            jump = x[i + 1] - x[i]
            r_dual = max(r_dual, abs(z) - li)
            r_cslack = max(r_cslack, min(li - abs(z), abs(jump)))
            r_sign = max(r_sign, -z * jump)
    return r_stat, r_dual, r_cslack, r_sign


def _solve(mat, rhs):
    """Gaussian elimination with partial pivoting; None if singular."""
    m = len(mat)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(mat)]
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-11:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(col + 1, m):
            f = aug[r][col] / aug[col][col]
            if f:
                for c in range(col, m + 1):
                    aug[r][c] -= f * aug[col][c]
    out = [0.0] * m
    for i in range(m - 1, -1, -1):
        s = aug[i][m]
        for j in range(i + 1, m):
            s -= aug[i][j] * out[j]
        out[i] = s / aug[i][i]
    return out


def brute_force(y, w, lam):
    """Exact optimum by enumerating jump sign patterns (tiny n only).

    For each pattern s in {-1,0,+1}^(n-1): edges with s_i = 0 force
    x_{i+1} = x_i and leave the dual z_i free; edges with s_i = +/-1 fix
    z_i = s_i*lam_i.  The KKT system is then linear; a pattern is
    consistent if its solution honours the prescribed signs/bounds.
    """
    n = len(y)
    best_obj = None
    best_x = None
    for signs in itertools.product((-1, 0, 1), repeat=n - 1):
        free = [e for e in range(n - 1) if signs[e] == 0]
        col = {e: n + j for j, e in enumerate(free)}
        m = n + len(free)
        mat = [[0.0] * m for _ in range(m)]
        rhs = [0.0] * m
        for i in range(n):                    # stationarity of x_i
            mat[i][i] = w[i]
            rhs[i] = w[i] * y[i]
            if i < n - 1:                     # -z_i term
                if signs[i] == 0:
                    mat[i][col[i]] -= 1.0
                else:
                    rhs[i] += signs[i] * lam[i]
            if i > 0:                         # +z_{i-1} term
                if signs[i - 1] == 0:
                    mat[i][col[i - 1]] += 1.0
                else:
                    rhs[i] -= signs[i - 1] * lam[i - 1]
        for r, e in enumerate(free):          # x_{e+1} - x_e = 0
            mat[n + r][e] = -1.0
            mat[n + r][e + 1] = 1.0
        sol = _solve(mat, rhs)
        if sol is None:
            continue
        xs = sol[:n]
        ok = True
        for e in range(n - 1):
            jump = xs[e + 1] - xs[e]
            if signs[e] == 1 and jump < -1e-7:
                ok = False
            elif signs[e] == -1 and jump > 1e-7:
                ok = False
            elif signs[e] == 0 and abs(sol[col[e]]) > lam[e] + 1e-7:
                ok = False
        if not ok:
            continue
        obj = objective(y, w, lam, xs)
        if best_obj is None or obj < best_obj - 1e-12:
            best_obj, best_x = obj, xs
    return best_obj, best_x


def rand_instance(rng, n):
    y = [rng.choice((rng.uniform(-10000, 10000),
                     rng.randint(-100, 100),
                     10000 * rng.choice((-1, 1)))) for _ in range(n)]
    w = [rng.choice((rng.uniform(0.1, 10), 0.1, 10,
                     rng.randint(1, 4))) for _ in range(n)]
    lam = [rng.choice((rng.uniform(0, 10000), 0.0, 10000,
                       rng.uniform(0, 3), rng.randint(0, 5)))
           for _ in range(max(n - 1, 0))]
    return y, w, lam


# ------------------------------------------------------------------ tests

def test_examples():
    x = tv.denoise([0, 4], [1, 3], [1])
    assert abs(x[0] - 1.0) < 1e-12 and abs(x[1] - 11.0 / 3.0) < 1e-12, x
    x = tv.denoise([0, 4], [1, 3], [0])
    assert x == [0.0, 4.0], x
    assert tv.denoise([], [], []) == []
    assert tv.denoise([7], [2], []) == [7.0]
    assert tv.denoise([-3.5], [0.1], []) == [-3.5]
    # zero penalties -> x == y
    y = [3.0, -2.5, 8.0, 0.0]
    assert tv.denoise(y, [1, 2, 3, 4], [0, 0, 0]) == y
    # constant y -> constant x
    x = tv.denoise([5.0] * 6, [1, 2, 3, 4, 5, 6], [1] * 5)
    assert all(abs(v - 5.0) < 1e-12 for v in x), x
    # huge penalties -> all equal to the weighted mean
    y = [0.0, 4.0, 1.0]
    w = [1.0, 3.0, 2.0]
    mean = sum(wi * yi for wi, yi in zip(w, y)) / sum(w)
    x = tv.denoise(y, w, [10000, 10000])
    assert all(abs(v - mean) < 1e-9 for v in x), (x, mean)
    # result is a full list of floats
    x = tv.denoise([0, 4], [1, 3], [1])
    assert isinstance(x, list) and len(x) == 2
    assert all(isinstance(v, float) for v in x)
    print("examples & edge cases: OK")


def test_random_kkt():
    rng = random.Random(20261114)
    worst = [0.0] * 4
    for n in (1, 2, 3, 5, 10, 50, 200, 1000, 5000):
        for _ in range(6 if n <= 200 else 3):
            y, w, lam = rand_instance(rng, n)
            x = tv.denoise(y, w, lam)
            assert len(x) == n and all(isinstance(v, float) for v in x)
            res = kkt_residuals(y, w, lam, x)
            worst = [max(a, b) for a, b in zip(worst, res)]
            tol = 1e-6 * (1 + max(abs(v) for v in y))
            assert all(r <= tol * 10 for r in res), (n, res, tol)
    print("random KKT residuals (stationarity, dual feas., compl. slack,"
          " sign): max =", ["%.3g" % v for v in worst])


def test_bruteforce():
    rng = random.Random(7050321)
    worst_obj = worst_arg = 0.0
    for n in (2, 3, 4, 5, 6):
        for _ in range(30):
            y, w, lam = rand_instance(rng, n)
            x = tv.denoise(y, w, lam)
            obj = objective(y, w, lam, x)
            bobj, bx = brute_force(y, w, lam)
            scale = 1.0 + abs(bobj)
            worst_obj = max(worst_obj, abs(obj - bobj) / scale)
            worst_arg = max(worst_arg,
                            max(abs(a - b) for a, b in zip(x, bx)))
            assert abs(obj - bobj) <= 1e-7 * scale, (n, obj, bobj)
            assert all(abs(a - b) <= 1e-4 * (1 + max(abs(v) for v in y))
                       for a, b in zip(x, bx)), (n, x, bx)
    print("brute-force cross-check (n<=6): max rel. obj. gap = %.3g, "
          "max |dx| = %.3g" % (worst_obj, worst_arg))


def test_validation():
    good = ([0.0, 4.0], [1.0, 3.0], [1.0])
    bad = [
        ((0, 4), [1, 3], [1]),                 # y not a list
        ([0, 4], (1, 3), [1]),                 # weights not a list
        ([0, 4], [1, 3], (1,)),                # penalties not a list
        ([0, 4], [1], [1]),                    # wrong len(weights)
        ([0, 4], [1, 3], []),                  # wrong len(penalties)
        ([0, 4], [1, 3], [1, 2]),              # wrong len(penalties)
        ([0, True], [1, 3], [1]),              # bool in y
        ([0, 4], [1, False], [1]),             # bool in weights
        ([0, 4], [1, 3], [True]),              # bool in penalties
        ([0, "4"], [1, 3], [1]),               # str in y
        ([0, float("nan")], [1, 3], [1]),      # NaN in y
        ([0, 4], [1, float("inf")], [1]),      # inf in weights
        ([0, 4], [1, 3], [float("nan")]),      # NaN in penalties
        ([0, 10001], [1, 3], [1]),             # |y| too large
        ([0, -10000.5], [1, 3], [1]),          # |y| too large
        ([0, 4], [0.099, 3], [1]),             # weight too small
        ([0, 4], [1, 10.001], [1]),            # weight too large
        ([0, 4], [1, 3], [-1e-9]),             # negative penalty
        ([0, 4], [1, 3], [10000.5]),           # penalty too large
        ([0.0] * 200001, [1.0] * 200001, [0.0] * 200000),  # n too large
    ]
    for args in bad:
        try:
            tv.denoise(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("no ValueError for %r" % (args,))
    # boundary values are accepted
    tv.denoise([-10000, 10000], [0.1, 10], [0])
    tv.denoise([-10000, 10000], [0.1, 10], [10000])
    # inputs are not mutated
    y, w, p = [0.0, 4.0, -1.5], [1.0, 3.0, 2.0], [1.0, 0.5]
    yc, wc, pc = y[:], w[:], p[:]
    tv.denoise(y, w, p)
    assert y == yc and w == wc and p == pc
    print("input validation & no-mutation: OK (%d illegal cases rejected)"
          % len(bad))


def main():
    test_examples()
    test_random_kkt()
    test_bruteforce()
    test_validation()
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
