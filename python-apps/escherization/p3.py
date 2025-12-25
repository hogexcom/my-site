"""
p3のエッシャー化行列を求める
"""

import numpy as np
import math
from scipy.linalg import null_space, norm
from typing import List, Tuple
import tiling_condition as tc

try:
    import pyvista as pv  # type: ignore
except Exception:  # pragma: no cover
    pv = None

# P3タイリング: シンプルな回転+並進

def trans_uv(boundary: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """P3タイリングの並進ベクトルu, vを計算
    
    Args:
        boundary: エッシャー化後の境界点 (N, 2)
        m: 1辺の頂点数
    
    Returns:
        (trans_u, trans_v): 並進ベクトルのタプル
    """
    # 0. 最初の頂点AとrotationIndicesの3番目の頂点Bを取得
    A = boundary[0].copy()
    rot_indices = rotation_indices(m)
    B = boundary[rot_indices[2]].copy()
    
    # 1. rotation_indicesの最初の頂点を中心に240度反時計回りに回転
    center1 = boundary[rot_indices[0]]
    angle1 = -2 * math.pi / 3  # 240度反時計回り = -120度
    
    # Aを回転してA'を得る
    dA = A - center1
    A_prime = np.array([
        math.cos(angle1) * dA[0] - math.sin(angle1) * dA[1],
        math.sin(angle1) * dA[0] + math.cos(angle1) * dA[1]
    ]) + center1
    
    # Bを回転してB'を得る
    dB = B - center1
    B_prime = np.array([
        math.cos(angle1) * dB[0] - math.sin(angle1) * dB[1],
        math.sin(angle1) * dB[0] + math.cos(angle1) * dB[1]
    ]) + center1
    
    # 2. B'を中心にA'を120度反時計回りに回転してA''を得る
    angle2 = 2 * math.pi / 3  # 120度反時計回り
    dA_prime = A_prime - B_prime
    A_double_prime = np.array([
        math.cos(angle2) * dA_prime[0] - math.sin(angle2) * dA_prime[1],
        math.sin(angle2) * dA_prime[0] + math.cos(angle2) * dA_prime[1]
    ]) + B_prime
    
    # 3. A''からAへのベクトルがtrans_u
    trans_u = A_double_prime - A
    
    # 4. trans_uを60度反時計回りに回転してtrans_vを得る
    angle3 = math.pi / 3  # 60度
    trans_v = np.array([
        math.cos(angle3) * trans_u[0] - math.sin(angle3) * trans_u[1],
        math.sin(angle3) * trans_u[0] + math.cos(angle3) * trans_u[1]
    ])
    
    return trans_u, trans_v

# 閉曲線の頂点数は以下の数の倍数である必要がある。
def p3_vertices_multiple():
    return 6


def number_of_vertices(m: int) -> int:
    return 6 * (m - 1)


def edges(m):
    last = 0
    edge_0 = [i for i in range(m)]
    last = m - 1
    edge_1 = [i for i in range(last, last + m)]
    last = last + m - 1
    edge_2 = [i for i in range(last, last + m)]
    last = last + m - 1
    edge_3 = [i for i in range(last, last + m)]
    last = last + m - 1
    edge_4 = [i for i in range(last, last + m)]
    last = last + m - 1
    edge_5 = [i for i in range(last, last + m)]
    edge_5[-1] = 0  # 最後の頂点は最初の頂点と同じ
    return edge_0, edge_1, edge_2, edge_3, edge_4, edge_5

def rotation_indices(m):
    edge_0, edge_1, edge_2, edge_3, edge_4, edge_5 = edges(m)
    return [edge_0[-1], edge_2[-1], edge_4[-1]]

# エッシャー化行列とその核を返す
# mは1辺あたりの頂点数
def kernel(m):
    xy_vec_len = number_of_vertices(m) * 2
    edge_0, edge_1, edge_2, edge_3, edge_4, edge_5 = edges(m)
    # print(edge_0, edge_1, edge_2, edge_3, edge_4, edge_5)
    # pass
    # print("xy_vec_len:", xy_vec_len)
    # row_num = (m - 1) * 6 * 2
    # (
    #     (
    #         (m - 1) * 3 # m - 1 個の３本のエッジがそれぞれ対応するエッジに写る分
    #           - 6 # 回転の角の点を条件から除く分
    #     ) * 2 # x,y成分分
    # )
    # print("row_num:", row_num)
    A = np.zeros((xy_vec_len, xy_vec_len))
    b = np.zeros((xy_vec_len,))
    x_p = np.zeros((xy_vec_len,))
    theta = math.pi + math.pi / 3

    tc.eq_rot(theta, edge_0[:-1], edge_1[1:][::-1], edge_0[-1], A, b)
    tc.eq_rot(theta, edge_2[:-1], edge_3[1:][::-1], edge_2[-1], A, b)
    tc.eq_rot(theta, edge_4[:-1], edge_5[1:][::-1], edge_4[-1], A, b)
    # print(edge_0[1:-1], edge_1[1:-1][::-1])

    # def update_A(row, start, base, end):
    #     A[row][x_index(start) - 1] = math.cos(theta)
    #     A[row][y_index(start) - 1] = -1 * math.sin(theta)
    #     A[row][x_index(base) - 1] = 1 - math.cos(theta)
    #     A[row][y_index(base) - 1] = math.sin(theta)
    #     A[row][x_index(end) - 1] = -1
    #     A[row + 1][x_index(start) - 1] = math.sin(theta)
    #     A[row + 1][y_index(start) - 1] = math.cos(theta)
    #     A[row + 1][x_index(base) - 1] = -1 * math.sin(theta)
    #     A[row + 1][y_index(base) - 1] = 1 - math.cos(theta)
    #     A[row + 1][y_index(end) - 1] = -1

    # row = 0
    # for L in range(0, 3):
    #     for i in range(1, m):
    #         update_A(row, 2 * m * L + i, 2 * m * L + m, 2 * m * L + 2 * m - i)
    #         row += 2
    #     k = (L + 1) * 2
    #     update_A(row, k * m, (k + 1) * m, (k + 2) * m)
    #     row += 2

    # print(A)
    G = null_space(A)
    return (A, G, b, x_p)


def test():
    m = 10
    A, G, b, x_p = kernel(m)
    xx = []
    yy = []
    for i in range(6):
        a = -math.pi + math.pi / 3 + (math.pi / 3) * (i + 0)
        b = a + (math.pi / 3)
        x = np.linspace(
            math.cos(a),
            math.cos(b),
            m,
        )[:-1] + np.sin(np.linspace(0, math.pi, m))[:-1] * 0.2
        y = np.linspace(
            math.sin(a),
            math.sin(b),
            m,
        )[:-1] + np.sin(np.linspace(0, math.pi, m))[:-1] * 0.1
        xx.extend(x)
        yy.extend(y)

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

    w = xy
    u_ = G @ (np.transpose(G) @ w)
    u = u_ + x_p
    redisual = A @ u - b
    print("A @ u - b norm is expected to be close to 0:", norm(redisual))
    xx = u[0::2]
    yy = u[1::2]
    
    # エッシャー化された境界を取得
    escherized_boundary = u.reshape(len(xx), 2)

    edge_0, edge_1, edge_2, edge_3, edge_4, edge_5 = edges(m)

    # エッシャー化後の各エッジの座標を取得
    u_edge_0 = np.array([[xx[i], yy[i]] for i in edge_0])
    u_edge_1 = np.array([[xx[i], yy[i]] for i in edge_1])
    u_edge_2 = np.array([[xx[i], yy[i]] for i in edge_2])
    u_edge_3 = np.array([[xx[i], yy[i]] for i in edge_3])
    u_edge_4 = np.array([[xx[i], yy[i]] for i in edge_4])
    u_edge_5 = np.array([[xx[i], yy[i]] for i in edge_5])

    theta = math.pi + math.pi / 3
    # 検証1: rot(edge_0) == reversed(edge_1)
    rot_edge_0 = (
        np.array(
            [
                [
                    math.cos(theta) * p[0] - math.sin(theta) * p[1],
                    math.sin(theta) * p[0] + math.cos(theta) * p[1],
                ]
                for p in u_edge_0 - u_edge_0[-1]
            ]
        )
        + u_edge_0[-1]
    )
    reversed_edge_1 = u_edge_1[::-1]
    error1 = np.linalg.norm(rot_edge_0 - reversed_edge_1)
    print(f"\n検証1: rot(edge_0) == reversed(edge_1)")
    print(f"  誤差: {error1:.6e}")

    # 検証2: rot(edge_2) == reversed(edge_3)
    rot_edge_2 = (
        np.array(
            [
                [
                    math.cos(theta) * p[0] - math.sin(theta) * p[1],
                    math.sin(theta) * p[0] + math.cos(theta) * p[1],
                ]
                for p in u_edge_2 - u_edge_2[-1]
            ]
        )
        + u_edge_2[-1]
    )
    reversed_edge_3 = u_edge_3[::-1]
    error2 = np.linalg.norm(rot_edge_2 - reversed_edge_3)
    print(f"検証2: rot(edge_2) == reversed(edge_3)")
    print(f"  誤差: {error2:.6e}")

    # 検証3: rot(edge_4) == reversed(edge_5)
    rot_edge_4 = (
        np.array(
            [
                [
                    math.cos(theta) * p[0] - math.sin(theta) * p[1],
                    math.sin(theta) * p[0] + math.cos(theta) * p[1],
                ]
                for p in u_edge_4 - u_edge_4[-1]
            ]
        )
        + u_edge_4[-1]
    )
    reversed_edge_5 = u_edge_5[::-1]
    error3 = np.linalg.norm(rot_edge_4 - reversed_edge_5)
    print(f"検証3: rot(edge_4) == reversed(edge_5)")
    print(f"  誤差: {error3:.6e}")

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

    # 残差と誤差をタイトルに表示
    ax[0].set_title("Original Boundary")
    ax[1].set_title(
        f"Escherized Boundary\nResidue: {norm(redisual):.2e}\nError1: {error1:.2e}, Error2: {error2:.2e}, Error3: {error3:.2e}"
    )
    
    # タイリング表示を追加
    visualize_p3_matplotlib_tiling(escherized_boundary, m, ax[2])

    pyplot.tight_layout()
    pyplot.show()
    
    # PyVistaタイリング表示を追加
    print("\n=== PyVistaタイリング表示 ===")
    # ダミーのメッシュデータを作成
    from triangulate_mesh import triangulate_boundary
    vertices, faces, boundary_indices = triangulate_boundary(escherized_boundary)
    
    # UV座標を0-1の範囲に正規化（各軸を独立に）
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    range_coords = max_coords - min_coords
    
    uv_coords = np.zeros((len(vertices), 2))
    if range_coords[0] > 0:
        uv_coords[:, 0] = (vertices[:, 0] - min_coords[0]) / range_coords[0]
    if range_coords[1] > 0:
        uv_coords[:, 1] = (vertices[:, 1] - min_coords[1]) / range_coords[1]
    
    # ダミー画像（グラデーション）
    image_rgb = np.ones((200, 200, 3)) * 0.8
    for i in range(200):
        for j in range(200):
            image_rgb[i, j, 2] = np.sin(i/20) * np.cos(j/20) * 0.5 + 0.5  # 青の波模様
    
    visualize_p3_pyvista_tiling(
        vertices,
        faces,
        uv_coords,
        image_rgb,
        escherized_boundary,
        m,
        "P3 Tiling with Texture",
        nx=10,
        ny=5,
        show_boundary=True,
    )


def debug():
    kernel(5)
    # print(p3_number_of_vertices(5))


def visualize_p3_matplotlib_tiling(escherized_boundary, m, ax):
    """MatplotlibでP3タイリングを表示（境界線のみ）
    
    Args:
        escherized_boundary: エッシャー化された境界点 (6*(m-1), 2)
        m: 1辺の頂点数
        ax: matplotlib Axes object
    """
    # 中央の境界線
    bx = np.append(escherized_boundary[:, 0], escherized_boundary[0, 0])
    by = np.append(escherized_boundary[:, 1], escherized_boundary[0, 1])
    ax.plot(bx, by, "g-", linewidth=2)

    # 回転タイル
    # P3の回転中心は境界上の3点
    edge_0, edge_1, edge_2, edge_3, edge_4, edge_5 = edges(m)
    rotation_indices = [edge_0[-1], edge_2[-1], edge_4[-1]]
    
    for i in range(1, 3):
        for rot_idx in rotation_indices:
            r = math.pi / 3 * 4 * i  # 240度, 480度 (= 120度)
            # 回転中心座標を取得
            cx = escherized_boundary[rot_idx, 0]
            cy = escherized_boundary[rot_idx, 1]
            dx = escherized_boundary[:, 0] - cx
            dy = escherized_boundary[:, 1] - cy
            rotated_x = np.cos(r) * dx - np.sin(r) * dy + cx
            rotated_y = np.sin(r) * dx + np.cos(r) * dy + cy

            # 閉曲線として描画
            rotated_x_closed = np.append(rotated_x, rotated_x[0])
            rotated_y_closed = np.append(rotated_y, rotated_y[0])
            ax.plot(rotated_x_closed, rotated_y_closed, "g-", linewidth=2, alpha=0.7)

    ax.set_title("P3 Tiling - Boundary Only (7 tiles)")
    ax.set_aspect("equal")


def visualize_p3_pyvista_tiling(
    deformed_vertices,
    faces,
    uv_coords,
    image_rgb,
    escherized_boundary,
    m,
    title="P3 Tiling with Texture",
    nx=2,
    ny=2,
    show_boundary=True,
):
    """PyVistaでP3タイリングされたテクスチャ付きメッシュを表示
    
    Args:
        deformed_vertices: 変形後の頂点座標 (V, 2)
        faces: 三角形インデックス (F, 3)
        uv_coords: UV座標 (V, 2)
        image_rgb: RGB画像 (H, W, 3), 0-1の範囲
        escherized_boundary: エッシャー化された境界点 (6*(m-1), 2)
        m: 1辺の頂点数
        title: 表示タイトル
        nx: u方向の並進回数
        ny: v方向の並進回数
        show_boundary: 境界線を表示するかどうか (デフォルト: True)
    """
    print(f"P3タイリング表示: {len(deformed_vertices)}頂点, {len(faces)}面")

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

    # P3対称性の回転中心点（境界上の特定の3点）
    edge_0, edge_1, edge_2, edge_3, edge_4, edge_5 = edges(m)
    rot_indices = rotation_indices(m)  # [edge_0[-1], edge_2[-1], edge_4[-1]]
    
    # PyVista用のメッシュ作成のヘルパー関数
    def create_mesh(vertices_2d):
        vertices_3d = np.column_stack([vertices_2d, np.zeros(len(vertices_2d))])
        faces_pv = np.column_stack([np.full(len(faces), 3), faces]).flatten()
        mesh = pv.PolyData(vertices_3d, faces_pv)
        # 元のUV座標をそのまま使用（頂点のインデックスは変わらないため）
        mesh.active_texture_coordinates = uv_coords
        return mesh

    # 頂点を指定の中心で回転させるヘルパー関数
    def rotate_vertices(vertices_2d, center_idx, angle):
        cx = escherized_boundary[center_idx, 0]
        cy = escherized_boundary[center_idx, 1]
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx, dy = vertices_2d[:, 0] - cx, vertices_2d[:, 1] - cy
        rotated_x = cos_a * dx - sin_a * dy + cx
        rotated_y = sin_a * dx + cos_a * dy + cy
        return np.column_stack([rotated_x, rotated_y])

    # 新しいタイリングロジック
    print("  新しいP3タイリングロジック開始")
    
    # ステップ1: 元のタイル（回転なし）
    base_tiles = []  # (vertices, boundary)のリスト
    base_tiles.append((deformed_vertices.copy(), escherized_boundary.copy()))
    print(f"    ステップ1: 元のタイル配置完了 (1個)")
    
    # ステップ2: 最初と次の頂点を中心に回転
    # 1つ目: 240度反時計回り（-120度）
    angle1 = -2 * math.pi / 3
    center_idx1 = rot_indices[0]
    rotated_verts1 = rotate_vertices(deformed_vertices, center_idx1, angle1)
    rotated_boundary1 = rotate_vertices(escherized_boundary, center_idx1, angle1)
    base_tiles.append((rotated_verts1, rotated_boundary1))
    
    # 2つ目: 120度反時計回り
    angle2 = 2 * math.pi / 3
    center_idx2 = rot_indices[1]
    rotated_verts2 = rotate_vertices(deformed_vertices, center_idx2, angle2)
    rotated_boundary2 = rotate_vertices(escherized_boundary, center_idx2, angle2)
    base_tiles.append((rotated_verts2, rotated_boundary2))
    
    print(f"    ステップ2: 回転タイル配置完了 (計3個)")
    
    # ステップ3: 3タイルを2方向に並進
    u_vec, v_vec = trans_uv(escherized_boundary, m)
    
    print(f"    並進ベクトル u: ({u_vec[0]:.4f}, {u_vec[1]:.4f}), 長さ: {np.linalg.norm(u_vec):.4f}")
    print(f"    並進ベクトル v: ({v_vec[0]:.4f}, {v_vec[1]:.4f}), 長さ: {np.linalg.norm(v_vec):.4f}")
    
    tile_count = 0
    
    # nx × ny の範囲で並進
    for i in range(nx):
        for j in range(ny):
            translation = i * u_vec + j * v_vec
            
            # 3つの基本タイルを並進
            for tile_verts, tile_boundary in base_tiles:
                translated_verts = tile_verts + translation
                
                # メッシュを追加
                mesh = create_mesh(translated_verts)
                plotter.add_mesh(mesh, texture=texture, show_edges=False, opacity=0.9)
                
                tile_count += 1
    
    print(f"    ステップ3: 並進完了 (計{tile_count}個のタイル)")

    print(f"  最終P3タイリング完了: {tile_count}個のタイル")

    # カメラを2D表示用に設定
    plotter.view_xy()
    plotter.camera.parallel_projection = True
    plotter.camera.zoom(0.5)

    plotter.add_title(f"{title}\n{tile_count} tiles total", font_size=12)
    plotter.set_background("white")

    print(f"  P3タイリング完了: {tile_count}個のタイル")
    print(
        "  表示オプション: マウスドラッグ=回転, Shift+ドラッグ=パン, ホイール=ズーム, q=終了"
    )

    plotter.show()


if __name__ == "__main__":
    from matplotlib import pyplot
    import pyvista as pv
    test()
    # debug()
