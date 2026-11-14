"""Exact 1-D weighted total-variation (TV) denoising / fused lasso.

``tv.denoise(y, weights, penalties)`` returns the unique minimizer ``x`` of

    f(x) = sum_i weights[i] * (x[i] - y[i])**2 / 2
         + sum_i penalties[i] * |x[i+1] - x[i]|

with ``n = len(y)``, ``len(weights) == n`` and
``len(penalties) == max(n-1, 0)``.  Because every ``weights[i] >= 0.1 > 0``
the objective is strongly convex, hence the minimizer is unique.

Algorithm (state and complexity)
--------------------------------
Dynamic programming over the chain, in the spirit of Johnson (2013),
"A dynamic programming algorithm for the fused lasso and L0-segmentation".

Let ``F_k(t)`` be the optimal cost of the first ``k+1`` samples with
``x[k] = t``.  ``F_k`` is convex and piecewise quadratic, so its derivative
``phi_k = F_k'`` is continuous, strictly increasing and piecewise linear.
The Bellman recursion

    F_{k+1}(t) = w[k+1] * (t - y[k+1])**2 / 2
               + min_s ( F_k(s) + penalties[k] * |t - s| )

becomes, on derivatives (the min-convolution is a Moreau envelope, whose
derivative is a clip):

    phi_{k+1}(t) = clip(phi_k(t), -penalties[k], +penalties[k])
                   + w[k+1] * (t - y[k+1]).

State maintained during the sweep
---------------------------------
* ``knots``: a deque of ``[location, stored_value, stored_slope]`` records
  describing the stored part of ``phi`` as a sorted piecewise-linear
  function; knots are only ever removed from either end (clips) and added
  at either end, so each of the <= 2(n-1) knots is pushed and popped at
  most once.
* a lazy linear term: ``phi_true(t) = phi_stored(t) + A*t + B``, so that
  adding ``w*(t - y)`` to ``phi`` costs O(1) (``A += w; B -= w*y``);
* ``s0`` / ``c0``: slope of the leftmost piece and an anchor value used
  when the knot list is empty;
* ``tminus[k]``, ``tplus[k]``: the two clip points where
  ``phi_k = -penalties[k]`` / ``+penalties[k]``, recorded for the backward
  substitution.

Invariant: every piece of ``phi_k`` has true slope >= w[k] >= 0.1 > 0, so
``phi_k`` is strictly increasing, every crossing exists and is unique, and
no division by zero can occur.

After the forward sweep, ``x[n-1]`` is the unique root of ``phi_{n-1}``
and ``x[k] = clip(x[k+1], tminus[k], tplus[k])`` (0-based edges).

Complexity
----------
Time:  O(n) amortized (each knot pushed/popped once) -- within the
       required O(n log(n+2)); no prefix is ever re-scanned and no dense
       matrix is built.
Space: O(n) auxiliary (knot deque, the two crossing arrays and ``x``).
"""

from collections import deque
import math

__all__ = ["denoise"]

_MAX_N = 200000
_Y_ABS = 10000.0
_W_LO, _W_HI = 0.1, 10.0
_P_LO, _P_HI = 0.0, 10000.0


def _ok(v, lo, hi):
    """True iff v is a finite non-bool int/float inside [lo, hi]."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    if isinstance(v, float) and not math.isfinite(v):
        return False
    return lo <= v <= hi


def _validate(y, weights, penalties):
    if not (isinstance(y, list) and isinstance(weights, list)
            and isinstance(penalties, list)):
        raise ValueError("y, weights and penalties must be lists")
    n = len(y)
    if len(weights) != n:
        raise ValueError("len(weights) must equal len(y)")
    if len(penalties) != (n - 1 if n else 0):
        raise ValueError("len(penalties) must equal max(n-1, 0)")
    if n > _MAX_N:
        raise ValueError("len(y) must be <= 200000")
    for v in y:
        if not _ok(v, -_Y_ABS, _Y_ABS):
            raise ValueError("y elements must be finite numbers "
                             "with |y[i]| <= 10000")
    for v in weights:
        if not _ok(v, _W_LO, _W_HI):
            raise ValueError("weights elements must satisfy "
                             "0.1 <= weights[i] <= 10")
    for v in penalties:
        if not _ok(v, _P_LO, _P_HI):
            raise ValueError("penalties elements must satisfy "
                             "0 <= penalties[i] <= 10000")


def denoise(y, weights, penalties):
    """Return the unique TV-denoised vector as a list of n floats."""
    _validate(y, weights, penalties)
    n = len(y)
    if n == 0:
        return []
    if n == 1:
        return [float(y[0])]

    # ---- forward sweep: track phi_k = derivative of the cost-to-go ----
    knots = deque()                # [loc, stored_val, stored_slope]
    s0 = float(weights[0])         # stored slope of the leftmost piece
    c0 = -s0 * y[0]                # stored value at t=0 (used if knots empty)
    A = 0.0                        # lazy term: phi_true(t) = phi_stored(t)
    B = 0.0                        #                    + A*t + B
    tminus = [0.0] * (n - 1)
    tplus = [0.0] * (n - 1)

    for k in range(n - 1):
        L = float(penalties[k])
        # -- right clip: locate t+ with phi_true(t+) = +L; drop knots >= L
        while knots:
            kn = knots[-1]
            if kn[1] + A * kn[0] + B < L:
                break
            knots.pop()
            if not knots:          # fell off the left end: re-anchor
                c0 = kn[1] - s0 * kn[0]
        if knots:
            kn = knots[-1]
            tp = (L - B - kn[1] + kn[2] * kn[0]) / (kn[2] + A)
        else:
            tp = (L - B - c0) / (s0 + A)
        # -- left clip: locate t- with phi_true(t-) = -L; drop knots <= -L
        while knots:
            kn = knots[0]
            if kn[1] + A * kn[0] + B > -L:
                break
            knots.popleft()
            s0 = kn[2]
            if not knots:          # fell off the right end: re-anchor
                c0 = kn[1] - kn[2] * kn[0]
        if knots:
            kn = knots[0]
            tm = (-L - B - kn[1] + s0 * kn[0]) / (s0 + A)
        else:
            tm = (-L - B - c0) / (s0 + A)
        tminus[k] = tm
        tplus[k] = tp
        # -- rebuild clipped phi: constant -L below tm, +L above tp
        knots.appendleft([tm, -L - A * tm - B, s0])
        knots.append([tp, L - A * tp - B, -A])
        s0 = -A
        # -- add w[k+1] * (t - y[k+1]) to obtain phi_{k+1}
        wk = float(weights[k + 1])
        A += wk
        B -= wk * y[k + 1]

    # ---- x[n-1] = unique root of phi_{n-1} ----
    if not knots:
        xn = -(c0 + B) / (s0 + A)
    else:
        xn = None
        prev = None
        for kn in knots:
            if kn[1] + A * kn[0] + B >= 0.0:
                if prev is None:
                    xn = (-B - kn[1] + s0 * kn[0]) / (s0 + A)
                else:
                    xn = (-B - prev[1] + prev[2] * prev[0]) / (prev[2] + A)
                break
            prev = kn
        if xn is None:             # root lies in the rightmost piece
            kn = knots[-1]
            xn = (-B - kn[1] + kn[2] * kn[0]) / (kn[2] + A)

    # ---- backward substitution: x[k] = clip(x[k+1], tminus[k], tplus[k])
    x = [0.0] * n
    x[n - 1] = xn
    for k in range(n - 2, -1, -1):
        v = x[k + 1]
        lo = tminus[k]
        hi = tplus[k]
        if v < lo:
            v = lo
        elif v > hi:
            v = hi
        x[k] = v
    return x
