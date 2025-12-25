#!/usr/bin/env python3
"""
Matplotlib Visualization for Escherization

エッシャー化プロセスのMatplotlibによる可視化モジュール
境界線、誤差グラフ、メッシュの2D描画を提供
"""

import numpy as np
from matplotlib import pyplot
import p3
import p1
import p2
from io import BytesIO
import base64


def _build_escherization_figure(
    X_best,
    best_idx,
    errors,
    best_error,
    escherized_boundary,
    vertices,
    faces,
    deformed_vertices,
    boundary_indices,
    tiling_pattern,
    m,
    n,
    tran_u=None,
    tran_v=None,
):
    """
    エッシャー化プロセスのMatplotlib可視化

    Args:
        X_best: 最適開始点で回転した元の境界 (n, 2)
        best_idx: 最適開始点のインデックス
        errors: 各開始点での誤差配列
        best_error: 最小誤差
        escherized_boundary: エッシャー化境界 (n, 2)
        vertices: メッシュ頂点（元の位置） (nv, 2)
        faces: メッシュ面 (nf, 3)
        deformed_vertices: 変形後のメッシュ頂点 (nv, 2)
        boundary_indices: 境界頂点のインデックス配列
        tiling_pattern: タイリングパターン ("P3", "P1", "P2")
        m: P3の1辺の頂点数
        n: P1/P2の1辺の頂点数
        tran_u: P1タイリング用の平行移動ベクトル1 (2,)
        tran_v: P1タイリング用の平行移動ベクトル2 (2,)

    Note:
        6パネル構成:
        1行目: 元の境界、誤差グラフ、エッシャー化境界
        2行目: 元のメッシュ、変形後メッシュ、タイリング
    """
    fig = pyplot.figure(figsize=(15, 10))

    # 1行目: 境界、誤差、エッシャー化境界
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_aspect("equal")
    xx = np.append(X_best[:, 0], X_best[0, 0])
    yy = np.append(X_best[:, 1], X_best[0, 1])
    ax1.plot(xx, yy, "b-", linewidth=2)
    ax1.plot(X_best[0, 0], X_best[0, 1], "ro", markersize=8)
    ax1.set_title(f"Original Boundary (Start: {best_idx})")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(errors, "b-")
    ax2.axvline(x=best_idx, color="r", linestyle="--", label=f"Best: {best_idx}")
    ax2.set_xlabel("Starting Point")
    ax2.set_ylabel("Error")
    ax2.set_title(f"Error vs Starting Point\nMin: {best_error:.1f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_aspect("equal")
    ux = np.append(escherized_boundary[:, 0], escherized_boundary[0, 0])
    uy = np.append(escherized_boundary[:, 1], escherized_boundary[0, 1])
    ax3.plot(ux, uy, "r-", linewidth=2)
    ax3.set_title("Escherized Boundary")
    ax3.grid(True, alpha=0.3)

    # 2行目: メッシュとタイリング（線画）
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_aspect("equal")
    ax4.triplot(vertices[:, 0], vertices[:, 1], faces, "b-", linewidth=0.5, alpha=0.7)
    ax4.set_title(f"Original Mesh\n({len(vertices)} verts)")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_aspect("equal")
    ax5.triplot(
        deformed_vertices[:, 0],
        deformed_vertices[:, 1],
        faces,
        "r-",
        linewidth=0.5,
        alpha=0.7,
    )
    # 境界線を別色で表示
    boundary_verts = deformed_vertices[boundary_indices]
    bx = np.append(boundary_verts[:, 0], boundary_verts[0, 0])
    by = np.append(boundary_verts[:, 1], boundary_verts[0, 1])
    ax5.plot(bx, by, "b-", linewidth=2, label="Boundary")
    ax5.set_title("Deformed Mesh")
    ax5.legend()

    # タイリング（境界線のみ）
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_aspect("equal")

    if tiling_pattern == "P3":
        p3.visualize_p3_matplotlib_tiling(escherized_boundary, m, ax6)
    elif tiling_pattern == "P1":
        if tran_u is None or tran_v is None:
            raise ValueError("P1タイリングにはtran_u, tran_vが必要です")
        p1.visualize_p1_matplotlib_tiling(
            escherized_boundary, tran_u, tran_v, ax6, nx=1, ny=1
        )
    elif tiling_pattern == "P2":
        p2.visualize_p2_matplotlib_tiling(escherized_boundary, m, n, ax6, nstages=2)

    pyplot.tight_layout()
    return fig


def visualize_escherization_matplotlib(
    X_best,
    best_idx,
    errors,
    best_error,
    escherized_boundary,
    vertices,
    faces,
    deformed_vertices,
    boundary_indices,
    tiling_pattern,
    m,
    n,
    tran_u=None,
    tran_v=None,
    show: bool = True,
):
    fig = _build_escherization_figure(
        X_best,
        best_idx,
        errors,
        best_error,
        escherized_boundary,
        vertices,
        faces,
        deformed_vertices,
        boundary_indices,
        tiling_pattern,
        m,
        n,
        tran_u,
        tran_v,
    )
    if show:
        pyplot.show()
    return fig


def render_escherization_matplotlib_png_base64(
    X_best,
    best_idx,
    errors,
    best_error,
    escherized_boundary,
    vertices,
    faces,
    deformed_vertices,
    boundary_indices,
    tiling_pattern,
    m,
    n,
    tran_u=None,
    tran_v=None,
    dpi: int = 150,
) -> str:
    fig = _build_escherization_figure(
        X_best,
        best_idx,
        errors,
        best_error,
        escherized_boundary,
        vertices,
        faces,
        deformed_vertices,
        boundary_indices,
        tiling_pattern,
        m,
        n,
        tran_u,
        tran_v,
    )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    pyplot.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_visualize_matplotlib():
    """
    Matplotlib可視化のテスト関数

    歪んだ六角形境界でP3タイリングをテスト
    """
    import math
    
    # P3パラメータ設定
    m = 21
    # P3に必要な境界点数: 6*m
    n_boundary_points = 6 * m
    
    # 歪んだ六角形の境界を生成（p3.pyのtest()を参考）
    xx = []
    yy = []
    radius = 100
    
    for i in range(6):
        # 各辺の始点と終点の角度
        a = -math.pi + math.pi / 3 + (math.pi / 3) * i
        b = a + (math.pi / 3)
        
        # 基本の直線辺
        x = np.linspace(math.cos(a), math.cos(b), m, endpoint=False) * radius
        y = np.linspace(math.sin(a), math.sin(b), m, endpoint=False) * radius
        
        # sin波形の変形を加える（外側に膨らむ）
        wave = np.sin(np.linspace(0, math.pi, m, endpoint=False))
        # 辺の外側方向を計算（90度回転）
        normal_x = -math.sin(a + math.pi / 6)
        normal_y = math.cos(a + math.pi / 6)
        
        x += wave * normal_x * 15  # 振幅を調整
        y += wave * normal_y * 15
        
        xx.extend(x)
        yy.extend(y)
    
    X_best = np.column_stack([xx, yy])

    # エッシャー化アルゴリズムを実際に適用
    from escherize_boundary import escherization
    escherized_boundary, X_rotated, best_idx, best_error, errors = escherization(
        X_best, "P3", m, m, resample=True
    )

    # メッシュ生成
    from triangulate_mesh import triangulate_boundary
    vertices, faces, boundary_indices = triangulate_boundary(X_rotated)

    # ARAP変形でエッシャー化境界に合わせる
    from mesh_deformation import solve_arap_deformation
    deformed_vertices = solve_arap_deformation(
        vertices, faces, boundary_indices, escherized_boundary
    )

    # 可視化
    visualize_escherization_matplotlib(
        X_rotated,
        best_idx,
        errors,
        best_error,
        escherized_boundary,
        vertices,
        faces,
        deformed_vertices,
        boundary_indices,
        tiling_pattern="P3",
        m=m,
        n=21,
    )


if __name__ == "__main__":
    # Pyodide用途では直接実行しない
    pass
