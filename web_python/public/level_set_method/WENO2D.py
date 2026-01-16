"""
Hamilton–Jacobi WENO (2D) sample implementation (HJ WENO5).

This module extends the 1D WENO5 reconstruction in WENO.py to 2D by
applying it independently along x-rows and y-columns on a uniform grid.

Core entry point:
    hj_weno5_derivatives_2d(phi, dx, dy, periodic=(True, True))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from WENO import WENODerivatives, hj_weno5_derivatives


FloatGrid = List[List[float]]


@dataclass(frozen=True)
class WENO2DDerivatives:
    phi_x_minus: FloatGrid
    phi_x_plus: FloatGrid
    phi_y_minus: FloatGrid
    phi_y_plus: FloatGrid


def _as_grid(phi: Iterable[Iterable[float]]) -> FloatGrid:
    return [[float(v) for v in row] for row in phi]


def _check_rectangular(phi: Sequence[Sequence[float]]) -> Tuple[int, int]:
    if not phi:
        return 0, 0
    cols = len(phi[0])
    for row in phi:
        if len(row) != cols:
            raise ValueError("phi must be a rectangular grid")
    return len(phi), cols


def hj_weno5_derivatives_2d(
    phi: Sequence[Sequence[float]],
    dx: float,
    dy: float,
    *,
    periodic: Tuple[bool, bool] = (True, True),
) -> WENO2DDerivatives:
    """
    Compute WENO5 approximations to (phi_x^-/phi_x^+) and (phi_y^-/phi_y^+)
    on a 2D uniform grid.

    Parameters
    ----------
    phi:
        Grid values phi[j][i] with shape (ny, nx).
    dx, dy:
        Uniform grid spacing in x and y directions.
    periodic:
        (periodic_x, periodic_y).
    """
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive")

    phi_grid = _as_grid(phi)
    ny, nx = _check_rectangular(phi_grid)
    if ny == 0 or nx == 0:
        return WENO2DDerivatives([], [], [], [])

    periodic_x, periodic_y = periodic

    if nx < 6 or ny < 6:
        raise ValueError("WENO5 needs at least 6 grid points in each direction")

    # X-derivatives: apply 1D WENO along each row.
    phi_x_minus: FloatGrid = [[0.0] * nx for _ in range(ny)]
    phi_x_plus: FloatGrid = [[0.0] * nx for _ in range(ny)]
    for j in range(ny):
        derivs: WENODerivatives = hj_weno5_derivatives(
            phi_grid[j], dx, periodic=periodic_x
        )
        phi_x_minus[j] = derivs.phi_x_minus
        phi_x_plus[j] = derivs.phi_x_plus

    # Y-derivatives: apply 1D WENO along each column.
    phi_y_minus: FloatGrid = [[0.0] * nx for _ in range(ny)]
    phi_y_plus: FloatGrid = [[0.0] * nx for _ in range(ny)]
    for i in range(nx):
        column = [phi_grid[j][i] for j in range(ny)]
        derivs = hj_weno5_derivatives(column, dy, periodic=periodic_y)
        for j in range(ny):
            phi_y_minus[j][i] = derivs.phi_x_minus[j]
            phi_y_plus[j][i] = derivs.phi_x_plus[j]

    return WENO2DDerivatives(
        phi_x_minus=phi_x_minus,
        phi_x_plus=phi_x_plus,
        phi_y_minus=phi_y_minus,
        phi_y_plus=phi_y_plus,
    )


def demo_constant_field() -> None:
    """
    Simple sanity check: constant field should yield near-zero derivatives.
    """
    nx, ny = 16, 12
    phi = [[1.0 for _ in range(nx)] for _ in range(ny)]
    derivs = hj_weno5_derivatives_2d(phi, dx=0.1, dy=0.2)
    max_abs = 0.0
    for grid in (derivs.phi_x_minus, derivs.phi_x_plus, derivs.phi_y_minus, derivs.phi_y_plus):
        for row in grid:
            for value in row:
                if abs(value) > max_abs:
                    max_abs = abs(value)
    print(f"max |derivative| = {max_abs:.3e}")


def _rectangle_signed_distance(
    x,
    y,
    center: Tuple[float, float],
    half_size: Tuple[float, float],
):
    cx, cy = center
    hx, hy = half_size
    dx = abs(x - cx) - hx
    dy = abs(y - cy) - hy
    return max(dx, dy)


def demo_advecting_rectangle(
    *,
    nx: int = 160,
    ny: int = 160,
    velocity: Tuple[float, float] = (0.5, 0.2),
    cfl: float = 0.5,
    steps: int = 200,
    steps_per_frame: int = 1,
    interval_ms: int = 40,
) -> None:
    """
    Advect a rectangle in the periodic domain and visualize corner preservation.
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("numpy and matplotlib are required for visualization")
        return

    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")

    lx, ly = 1.0, 1.0
    dx = lx / nx
    dy = ly / ny
    u, v = velocity

    if u == 0.0 and v == 0.0:
        raise ValueError("velocity must be non-zero")

    dt = cfl / (abs(u) / dx + abs(v) / dy)

    x = np.linspace(0.0, lx, nx, endpoint=False)
    y = np.linspace(0.0, ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y)

    center = (0.5 * lx, 0.5 * ly)
    half_size = (0.2 * lx, 0.15 * ly)
    phi = [
        [_rectangle_signed_distance(xi, yj, center, half_size) for xi in x]
        for yj in y
    ]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", "box")
    ax.set_xlim(0.0, lx)
    ax.set_ylim(0.0, ly)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    contour = ax.contour(X, Y, np.asarray(phi), levels=[0.0], colors="#1f77b4", linewidths=2.0)
    title = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
    )

    state = {"step": 0}

    def _rhs(phi_local):
        derivs = hj_weno5_derivatives_2d(phi_local, dx, dy, periodic=(True, True))
        phi_x = derivs.phi_x_minus if u > 0.0 else derivs.phi_x_plus
        phi_y = derivs.phi_y_minus if v > 0.0 else derivs.phi_y_plus
        return [
            [
                -(u * phi_x[j][i] + v * phi_y[j][i])
                for i in range(nx)
            ]
            for j in range(ny)
        ]

    def _axpy(a, x, y):
        return [
            [a * x[j][i] + y[j][i] for i in range(nx)]
            for j in range(ny)
        ]

    def update(_frame: int):
        nonlocal phi, contour
        for _ in range(steps_per_frame):
            k1 = _rhs(phi)
            phi1 = _axpy(dt, k1, phi)
            k2 = _rhs(phi1)
            phi2 = _axpy(dt, k2, phi1)
            phi2 = [
                [0.75 * phi[j][i] + 0.25 * phi2[j][i] for i in range(nx)]
                for j in range(ny)
            ]
            k3 = _rhs(phi2)
            phi3 = _axpy(dt, k3, phi2)
            phi = [
                [(1.0 / 3.0) * phi[j][i] + (2.0 / 3.0) * phi3[j][i] for i in range(nx)]
                for j in range(ny)
            ]
            state["step"] += 1
            if state["step"] >= steps:
                break

        t = state["step"] * dt
        Xs = (X - u * t) % lx
        Ys = (Y - v * t) % ly
        phi_exact = np.maximum(np.abs(Xs - center[0]) - half_size[0], np.abs(Ys - center[1]) - half_size[1])
        phi_arr = np.asarray(phi)
        error = float(np.sqrt(np.mean((phi_arr - phi_exact) ** 2)))

        for coll in contour.collections:
            coll.remove()
        contour = ax.contour(X, Y, phi_arr, levels=[0.0], colors="#1f77b4", linewidths=2.0)
        title.set_text(f"WENO5  t={t:.3f}  L2 error={error:.3e}")
        return contour.collections + [title]

    anim = FuncAnimation(
        fig,
        update,
        frames=max(1, (steps + steps_per_frame - 1) // steps_per_frame),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    fig._weno2d_anim = anim
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_constant_field()
    demo_advecting_rectangle()
