"""
p1のエッシャー化行列を求める
"""

import numpy as np
import math
from scipy.linalg import null_space, norm, lstsq
from typing import Dict, List, Tuple
import tiling_condition as tc

try:
    import pyvista as pv  # type: ignore
except Exception:  # pragma: no cover
    pv = None

# P1タイリング: シンプルな並進のみ

"""
tran_u, tran_v: 並進ベクトル。写像でもある
m: tran_u方向のエッジの頂点数
n: tran_v方向のエッジの頂点数
"""


def number_of_vertices(m: int, n: int) -> int:
    return 2 * (m - 1) + 2 * (n - 1)


def kernel(tran_u: Tuple[float, float], tran_v: Tuple[float, float], m: int, n: int):
    num_vertices = number_of_vertices(m, n)
    len_xy_vec = 2 * num_vertices
    # どのエッジがどのエッジに写るかで決まる。2はx,y成分の分
    row_num = (
        (m - 1) * 2  # edge 0 が edge 2 に写る分
        + (n - 1) * 2  # edge 1 が edge 3 に写る分
        + (m - 1) * 2  # edge 2 が edge 0 に写る分
        + (n - 1) * 2  # edge 3 が edge 1 に写る分
    )
    # print(num_vertices, len_xy_vec)
    A = np.zeros((row_num, len_xy_vec))
    b = np.zeros((row_num,))

    edge_0 = [i for i in range(m)]
    edge_1 = [i for i in range(m - 1, m - 1 + n)]
    edge_2 = [i for i in range(m + n - 2, 2 * m + n - 2)]
    edge_3 = [i for i in range(2 * m + n - 3, 2 * m + 2 * n - 3)]
    edge_3[-1] = 0  # 最後の頂点は最初の頂点と同じ

    # print(edge_0, edge_1, edge_2, edge_3)
    # print(A.shape, b.shape)
    # pass

    # def eqtran(tran, edge1, edge2):
    #     eq(tran, edge1, edge2)
    # eq(inv(tran), edge2, edge1)

    # print(edge_0, edge_0[::-1], edge_0)
    tc.eq_tran(tran_v, edge_0, edge_2[::-1], A, b)
    tc.eq_tran(tc.inv_tran(tran_u), edge_1, edge_3[::-1], A, b)

    # 最小二乗解（可解なら特殊解）
    x_p, residuals, rank, s = lstsq(A, b)

    # print("特殊解 x_p =", x_p)
    # print(A)
    G = null_space(A)
    return (A, G, b, x_p)


def debug():
    kernel((1, 0), (0, 1), 5, 4)


def test():
    tran_u = (1, 0.2)
    tran_v = (0.1, 1)
    m = 10
    n = 10
    A, G, b, x_p = kernel(tran_u, tran_v, m, n)
    xx = []
    yy = []
    edge_0_x = np.linspace(0, tran_u[0], m)
    edge_0_y = np.linspace(0, tran_u[1], m) + np.sin(np.linspace(0, math.pi, m)) * 0.2
    edge_1_x = (
        np.linspace(tran_u[0], tran_u[0] + tran_v[0], n)[1:]
        + np.sin(np.linspace(0, math.pi * 2, n))[1:] * 0.1
    )
    edge_1_y = np.linspace(tran_u[1], tran_u[1] + tran_v[1], n)[1:]
    edge_2_x = np.linspace(tran_u[0] + tran_v[0], tran_v[0], m)[1:]
    edge_2_y = (
        np.linspace(tran_u[1] + tran_v[1], tran_v[1], m)[1:]
        + np.sin(np.linspace(0, math.pi, m))[1:] * 0.2
    )
    edge_3_x = (
        np.linspace(tran_v[0], 0, n)[1:-1]
        + np.sin(np.linspace(math.pi * 2, 0, n)[1:-1]) * 0.1
    )
    edge_3_y = np.linspace(tran_v[1], 0, n)[1:-1]
    xx.extend(edge_0_x)
    yy.extend(edge_0_y)
    xx.extend(edge_1_x)
    yy.extend(edge_1_y)
    xx.extend(edge_2_x)
    yy.extend(edge_2_y)
    xx.extend(edge_3_x)
    yy.extend(edge_3_y)

    fix, ax = pyplot.subplots(1, 3, figsize=(18, 6))
    ax[0].set_aspect("equal")
    ax[0].plot(xx, yy)

    # 頂点に番号を振る（1から開始）
    for i in range(len(xx)):
        ax[0].annotate(
            str(i + 1),
            (xx[i], yy[i]),
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", alpha=0.7),
        )

    xy = np.transpose(np.array([xx, yy])).flatten()
    # print(xy.shape, A.shape)
    vec = A @ xy - b
    print("A @ xy norm is expected to be close to 0:", np.linalg.norm(vec))

    w = xy
    u_ = G @ (np.transpose(G) @ w)
    u = u_ + x_p
    redisual = A @ u - b
    print("A @ u - b norm is expected to be close to 0:", norm(redisual))
    xx = u[0::2]
    yy = u[1::2]
    
    # エッシャー化された境界を取得
    escherized_boundary = u.reshape(len(xx), 2)

    # エッシャー化後のedge0~3を取得
    edge_0 = [i for i in range(m)]
    edge_1 = [i for i in range(m - 1, m - 1 + n)]
    edge_2 = [i for i in range(m + n - 2, 2 * m + n - 2)]
    edge_3 = [i for i in range(2 * m + n - 3, 2 * m + 2 * n - 3)]
    edge_3[-1] = 0

    # エッシャー化後の各エッジの座標を取得
    u_edge_0 = np.array([[xx[i], yy[i]] for i in edge_0])
    u_edge_1 = np.array([[xx[i], yy[i]] for i in edge_1])
    u_edge_2 = np.array([[xx[i], yy[i]] for i in edge_2])
    u_edge_3 = np.array([[xx[i], yy[i]] for i in edge_3])

    # 検証1: tran_v(edge_0) = reversed(edge_2)
    tran_v_edge_0 = u_edge_0 + np.array(tran_v)
    reversed_edge_2 = u_edge_2[::-1]
    error1 = np.linalg.norm(tran_v_edge_0 - reversed_edge_2)
    print(f"\n検証1: tran_v(edge_0) == reversed(edge_2)")
    print(f"  誤差: {error1:.6e}")

    # 検証2: tran_u(edge_3) = reversed(edge_1)
    tran_u_edge_3 = u_edge_3 + np.array(tran_u)
    reversed_edge_1 = u_edge_1[::-1]
    error2 = np.linalg.norm(tran_u_edge_3 - reversed_edge_1)
    print(f"検証2: tran_u(edge_3) == reversed(edge_1)")
    print(f"  誤差: {error2:.6e}")

    ax[1].set_aspect("equal")
    ax[1].plot(xx, yy)
    # 頂点に番号を振る（1から開始）
    for i in range(len(xx)):
        ax[1].annotate(
            str(i + 1),
            (xx[i], yy[i]),
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", alpha=0.7),
        )
    
    # タイリング表示を追加
    visualize_p1_matplotlib_tiling(escherized_boundary, tran_u, tran_v, ax[2], nx=1, ny=1)
    
    pyplot.tight_layout()
    pyplot.show()
    
    # PyVistaタイリング表示を追加
    print("\n=== PyVistaタイリング表示 ===")
    # ダミーのメッシュデータを作成
    from triangulate_mesh import triangulate_boundary
    vertices, faces, boundary_indices = triangulate_boundary(escherized_boundary)
    
    # UV座標（簡易版）
    uv_coords = np.zeros((len(vertices), 2))
    for i in range(len(vertices)):
        uv_coords[i] = vertices[i] / (np.max(vertices, axis=0) + 1e-8)
    
    # ダミー画像（グラデーション）
    image_rgb = np.ones((200, 200, 3)) * 0.8
    for i in range(200):
        image_rgb[i, :, 0] = i / 200  # 赤のグラデーション
    
    visualize_p1_pyvista_tiling(
        vertices,
        faces,
        uv_coords,
        image_rgb,
        tran_u,
        tran_v,
        "P1 Tiling with Texture",
        nx=5,
        ny=5,
    )


def visualize_p1_matplotlib_tiling(escherized_boundary, tran_u, tran_v, ax, nx=1, ny=1):
    """MatplotlibでP1タイリングを表示（境界線のみ）
    
    Args:
        escherized_boundary: エッシャー化された境界点 (N, 2)
        tran_u: 並進ベクトルu (2,)
        tran_v: 並進ベクトルv (2,)
        ax: matplotlib Axes object
        nx: x方向のタイル数（デフォルト: 1, 合計3x3グリッド）
        ny: y方向のタイル数（デフォルト: 1, 合計3x3グリッド）
    """
    tran_u_vec = np.array(tran_u)
    tran_v_vec = np.array(tran_v)
    
    # (2*nx+1) x (2*ny+1) グリッドで表示
    for i in range(-nx, nx + 1):
        for j in range(-ny, ny + 1):
            translation = i * tran_u_vec + j * tran_v_vec
            translated_boundary = escherized_boundary + translation
            
            # 閉曲線として描画
            bx = np.append(translated_boundary[:, 0], translated_boundary[0, 0])
            by = np.append(translated_boundary[:, 1], translated_boundary[0, 1])
            ax.plot(bx, by, "g-", linewidth=2, alpha=0.7)
    
    ax.set_title(f"P1 Tiling - Boundary Only ({(2*nx+1)*(2*ny+1)} tiles)")
    ax.set_aspect("equal")


def visualize_p1_pyvista_tiling(
    deformed_vertices,
    faces,
    uv_coords,
    image_rgb,
    tran_u,
    tran_v,
    title="P1 Tiling with Texture",
    nx=3,
    ny=3,
):
    """PyVistaでP1タイリングされたテクスチャ付きメッシュを表示
    
    Args:
        deformed_vertices: 変形後の頂点座標 (V, 2)
        faces: 三角形インデックス (F, 3)
        uv_coords: UV座標 (V, 2)
        image_rgb: RGB画像 (H, W, 3), 0-1の範囲
        tran_u: 並進ベクトルu (2,)
        tran_v: 並進ベクトルv (2,)
        title: 表示タイトル
        nx: x方向のタイル数
        ny: y方向のタイル数
    """
    print(f"P1タイリング表示: {len(deformed_vertices)}頂点, {len(faces)}面")

    # テクスチャ画像を準備
    if image_rgb.dtype != np.uint8:
        tex_image = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
    else:
        tex_image = image_rgb

    if tex_image.shape[2] == 3:
        alpha = np.full(
            (tex_image.shape[0], tex_image.shape[1], 1), 255, dtype=np.uint8
        )
        tex_image = np.concatenate([tex_image, alpha], axis=2)

    texture = pv.numpy_to_texture(tex_image)

    # プロッター作成
    plotter = pv.Plotter(window_size=[1400, 1000])

    # PyVista用のメッシュ作成のヘルパー関数
    def create_mesh(vertices_2d):
        vertices_3d = np.column_stack([vertices_2d, np.zeros(len(vertices_2d))])
        faces_pv = np.column_stack([np.full(len(faces), 3), faces]).flatten()
        mesh = pv.PolyData(vertices_3d, faces_pv)
        mesh.active_texture_coordinates = uv_coords
        return mesh

    # シンプルな並進タイリング
    tran_u_vec = np.array(tran_u)
    tran_v_vec = np.array(tran_v)
    
    tile_count = 0
    
    # nx × ny の範囲でタイルを配置
    for i in range(nx):
        for j in range(ny):
            # 平行移動ベクトルを計算
            translation = i * tran_u_vec + j * tran_v_vec
            
            # 頂点を平行移動
            translated_verts = deformed_vertices + translation
            
            # メッシュを作成して追加
            mesh = create_mesh(translated_verts)
            plotter.add_mesh(mesh, texture=texture, show_edges=False, opacity=0.9)
            
            tile_count += 1

    print(f"  P1タイリング完了: {tile_count}個のタイル ({nx} × {ny} グリッド)")

    # カメラを2D表示用に設定
    plotter.view_xy()
    plotter.camera.parallel_projection = True
    plotter.camera.zoom(0.5)

    plotter.add_title(f"{title}\n{tile_count} tiles total", font_size=12)
    plotter.set_background("white")

    print(f"  P1タイリング完了: {tile_count}個のタイル")
    print(
        "  表示オプション: マウスドラッグ=回転, Shift+ドラッグ=パン, ホイール=ズーム, q=終了"
    )

    plotter.show()


if __name__ == "__main__":
    from matplotlib import pyplot
    import pyvista as pv

    # debug()
    test()
