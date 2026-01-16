"""
Hamilton–Jacobi WENO (1D) sample implementation (HJ WENO5).

Implements the 5th-order accurate HJ WENO reconstruction described by
Jiang & Peng / Jiang & Shu, in the form commonly presented in level set texts.

For (phi_x^-)i:
  v1 = (D^- phi)_{i-2}, v2 = (D^- phi)_{i-1}, v3 = (D^- phi)_i,
  v4 = (D^- phi)_{i+1}, v5 = (D^- phi)_{i+2}

Candidate 3rd-order approximations:
  phi_x^1 =  v1/3 - 7v2/6 + 11v3/6
  phi_x^2 = -v2/6 + 5v3/6 +  v4/3
  phi_x^3 =  v3/3 + 5v4/6 -  v5/6

Smoothness indicators (S1,S2,S3) and nonlinear weights (w1,w2,w3) follow
equations (3.32)–(3.41) in the excerpt. For (phi_x^+)i we use the mirrored
definition using forward differences.

Optional: numba for acceleration (falls back to pure Python).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sin, sqrt
from typing import Iterable, List, Sequence, Tuple

# from numba import njit
def njit(*_args, **_kwargs):
    def wrapper(func):
        return func
    return wrapper


FloatArray = List[float]


@dataclass(frozen=True)
class WENODerivatives:
    phi_x_minus: FloatArray
    phi_x_plus: FloatArray


def _as_floats(values: Iterable[float]) -> FloatArray:
    return [float(v) for v in values]


def _pad_periodic(phi: Sequence[float], pad: int) -> FloatArray:
    if pad <= 0:
        return list(phi)
    n = len(phi)
    if n == 0:
        return []
    return list(phi[n - pad :]) + list(phi) + list(phi[:pad])


def _l2_error(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("length mismatch")
    if not a:
        return 0.0
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return sqrt(s / len(a))


def _safe_square(x: float, limit: float = 1e154) -> float:
    if x > limit:
        x = limit
    elif x < -limit:
        x = -limit
    return x * x


@njit(cache=True)
def _hj_weno5_derivatives_core(phi_list: FloatArray, dx: float, periodic: bool) -> Tuple[FloatArray, FloatArray]:
    n = len(phi_list)
    pad = 3
    m = n + 2 * pad
    phi_pad = [0.0] * m

    if periodic:
        for i in range(pad):
            phi_pad[i] = phi_list[n - pad + i]
        for i in range(n):
            phi_pad[pad + i] = phi_list[i]
        for i in range(pad):
            phi_pad[pad + n + i] = phi_list[i]
        start = 0
        end = n
    else:
        for i in range(pad):
            phi_pad[i] = phi_list[0]
        for i in range(n):
            phi_pad[pad + i] = phi_list[i]
        for i in range(pad):
            phi_pad[pad + n + i] = phi_list[n - 1]
        start = 0
        end = n

    dm = [0.0] * m
    dp = [0.0] * m
    for i in range(1, m - 1):
        dm[i] = (phi_pad[i] - phi_pad[i - 1]) / dx
        dp[i] = (phi_pad[i + 1] - phi_pad[i]) / dx

    phi_x_minus = [0.0] * n
    phi_x_plus = [0.0] * n

    for i in range(start, end):
        ip = i + pad

        # ---- phi_x^- at i ----
        v1 = dm[ip - 2]
        v2 = dm[ip - 1]
        v3 = dm[ip]
        v4 = dm[ip + 1]
        v5 = dm[ip + 2]
        if not (isfinite(v1) and isfinite(v2) and isfinite(v3) and isfinite(v4) and isfinite(v5)):
            phi_x_minus[i] = 0.0
            phi_x_plus[i] = 0.0
            continue

        p1 = (1.0 / 3.0) * v1 - (7.0 / 6.0) * v2 + (11.0 / 6.0) * v3
        p2 = -(1.0 / 6.0) * v2 + (5.0 / 6.0) * v3 + (1.0 / 3.0) * v4
        p3 = (1.0 / 3.0) * v3 + (5.0 / 6.0) * v4 - (1.0 / 6.0) * v5

        s1 = (13.0 / 12.0) * _safe_square(v1 - 2.0 * v2 + v3) + (1.0 / 4.0) * _safe_square(v1 - 4.0 * v2 + 3.0 * v3)
        s2 = (13.0 / 12.0) * _safe_square(v2 - 2.0 * v3 + v4) + (1.0 / 4.0) * _safe_square(v2 - v4)
        s3 = (13.0 / 12.0) * _safe_square(v3 - 2.0 * v4 + v5) + (1.0 / 4.0) * _safe_square(3.0 * v3 - 4.0 * v4 + v5)

        vmax = v1 * v1
        if v2 * v2 > vmax:
            vmax = v2 * v2
        if v3 * v3 > vmax:
            vmax = v3 * v3
        if v4 * v4 > vmax:
            vmax = v4 * v4
        if v5 * v5 > vmax:
            vmax = v5 * v5
        eps = 1e-6 * vmax + 1e-20

        try:
            a1 = 0.1 / ((s1 + eps) ** 2)
        except OverflowError:
            a1 = 0.0
        try:
            a2 = 0.6 / ((s2 + eps) ** 2)
        except OverflowError:
            a2 = 0.0
        try:
            a3 = 0.3 / ((s3 + eps) ** 2)
        except OverflowError:
            a3 = 0.0
        asum = a1 + a2 + a3
        if asum == 0.0:
            w1 = 1.0 / 3.0
            w2 = 1.0 / 3.0
            w3 = 1.0 / 3.0
        else:
            w1 = a1 / asum
            w2 = a2 / asum
            w3 = a3 / asum
        phi_x_minus[i] = w1 * p1 + w2 * p2 + w3 * p3

        # ---- phi_x^+ at i ----
        v1 = dp[ip + 2]
        v2 = dp[ip + 1]
        v3 = dp[ip]
        v4 = dp[ip - 1]
        v5 = dp[ip - 2]

        p1 = (1.0 / 3.0) * v1 - (7.0 / 6.0) * v2 + (11.0 / 6.0) * v3
        p2 = -(1.0 / 6.0) * v2 + (5.0 / 6.0) * v3 + (1.0 / 3.0) * v4
        p3 = (1.0 / 3.0) * v3 + (5.0 / 6.0) * v4 - (1.0 / 6.0) * v5

        s1 = (13.0 / 12.0) * _safe_square(v1 - 2.0 * v2 + v3) + (1.0 / 4.0) * _safe_square(v1 - 4.0 * v2 + 3.0 * v3)
        s2 = (13.0 / 12.0) * _safe_square(v2 - 2.0 * v3 + v4) + (1.0 / 4.0) * _safe_square(v2 - v4)
        s3 = (13.0 / 12.0) * _safe_square(v3 - 2.0 * v4 + v5) + (1.0 / 4.0) * _safe_square(3.0 * v3 - 4.0 * v4 + v5)

        vmax = v1 * v1
        if v2 * v2 > vmax:
            vmax = v2 * v2
        if v3 * v3 > vmax:
            vmax = v3 * v3
        if v4 * v4 > vmax:
            vmax = v4 * v4
        if v5 * v5 > vmax:
            vmax = v5 * v5
        eps = 1e-6 * vmax + 1e-20

        try:
            a1 = 0.1 / ((s1 + eps) ** 2)
        except OverflowError:
            a1 = 0.0
        try:
            a2 = 0.6 / ((s2 + eps) ** 2)
        except OverflowError:
            a2 = 0.0
        try:
            a3 = 0.3 / ((s3 + eps) ** 2)
        except OverflowError:
            a3 = 0.0
        asum = a1 + a2 + a3
        if asum == 0.0:
            w1 = 1.0 / 3.0
            w2 = 1.0 / 3.0
            w3 = 1.0 / 3.0
        else:
            w1 = a1 / asum
            w2 = a2 / asum
            w3 = a3 / asum
        phi_x_plus[i] = w1 * p1 + w2 * p2 + w3 * p3

    return phi_x_minus, phi_x_plus


def hj_weno5_derivatives(phi: Sequence[float], dx: float, *, periodic: bool = True) -> WENODerivatives:
    """
    Compute HJ WENO5 approximations to (phi_x^-)i and (phi_x^+)i on a 1D uniform grid.

    Parameters
    ----------
    phi:
        Values phi_i on grid nodes x_i.
    dx:
        Uniform grid spacing.
    periodic:
        If True, use periodic padding so all points are treated uniformly.
        If False, only the interior points are computed and boundary values are 0.0.
    """
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    phi_list = _as_floats(phi)
    n = len(phi_list)
    if n == 0:
        return WENODerivatives([], [])
    if n < 6:
        raise ValueError("WENO5 needs at least 6 grid points")

    phi_x_minus, phi_x_plus = _hj_weno5_derivatives_core(phi_list, dx, periodic)
    return WENODerivatives(phi_x_minus=phi_x_minus, phi_x_plus=phi_x_plus)


def linear_advection_rhs_weno5(phi: Sequence[float], dx: float, u: float, *, periodic: bool = True) -> FloatArray:
    """
    Semi-discrete RHS for linear advection: phi_t + u * phi_x = 0, using WENO5 in space.
    """
    d = hj_weno5_derivatives(phi, dx, periodic=periodic)
    if u > 0.0:
        return [-u * p for p in d.phi_x_minus]
    if u < 0.0:
        return [-u * p for p in d.phi_x_plus]
    return [0.0 for _ in phi]


def demo() -> None:
    """
    Console demo:
      - Smooth function accuracy (WENO5 vs ENO3)
      - Kink behavior sample near x=0.5 for phi(x)=|x-0.5|
    """
    print("=== HJ WENO5 demo (1D, periodic) ===")

    # smooth test
    n = 200
    dx = 1.0 / n
    x = [(i + 0.5) * dx for i in range(n)]
    phi = [sin(2.0 * pi * xi) for xi in x]
    exact = [2.0 * pi * cos(2.0 * pi * xi) for xi in x]

    d_w = hj_weno5_derivatives(phi, dx, periodic=True)
    approx_w = [(a + b) * 0.5 for a, b in zip(d_w.phi_x_minus, d_w.phi_x_plus)]
    print(f"smooth: WENO5  L2 error={_l2_error(approx_w, exact):.3e}")

    try:
        from ENO import hj_eno_derivatives  # type: ignore
    except ImportError:
        hj_eno_derivatives = None

    if hj_eno_derivatives is not None:
        d_e = hj_eno_derivatives(phi, dx, order=3, periodic=True)
        approx_e = [(a + b) * 0.5 for a, b in zip(d_e.phi_x_minus, d_e.phi_x_plus)]
        print(f"smooth: ENO3   L2 error={_l2_error(approx_e, exact):.3e}")

    # kink test
    print("\n--- kink sample (phi(x)=|x-0.5|) ---")
    phi2 = [abs(xi - 0.5) for xi in x]
    d2 = hj_weno5_derivatives(phi2, dx, periodic=True)
    i0 = n // 2
    for j in range(i0 - 3, i0 + 4):
        jj = j % n
        print(
            f"i={jj:4d} x={x[jj]:.3f}  phi={phi2[jj]:.3f}  "
            f"phi_x-={d2.phi_x_minus[jj]: .3f}  phi_x+={d2.phi_x_plus[jj]: .3f}"
        )


if __name__ == "__main__":
    demo()
