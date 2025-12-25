"""
p2のエッシャー化行列を求める
"""

import numpy as np
import math
from scipy.linalg import null_space, norm, lstsq
from typing import Dict, List, Tuple
import sys
import tiling_condition as tc

try:
    import pyvista as pv  # type: ignore
except Exception:  # pragma: no cover
    pv = None

# P2タイリング: シンプルな回転+並進

"""
tran_u, tran_v: 並進ベクトル。写像でもある
m: tran_u方向のエッジの頂点数
n: tran_v方向のエッジの頂点数
"""


def trans_uv(boundary: np.ndarray, m: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """P2タイリングの並進ベクトルu, vを計算
    
    Args:
        boundary: エッシャー化後の境界点 (N, 2)
        m: tran_u方向のエッジの頂点数
        n: tran_v方向のエッジの頂点数
    
    Returns:
        (trans_u, trans_v): 並進ベクトルのタプル
    """
    edge_0, edge_1, edge_2, edge_3 = edges(m, n)
    center_0, center_1, center_2, center_3 = center_of_edges(m, n)
    
    # trans_uの計算
    # 1. 辺0の最初の点Aと辺3の中点Bを取得
    A = boundary[edge_0[0]].copy()
    B = boundary[center_3].copy()
    
    # 2. 辺1の中点で180度回転させてA'とB'を得る
    center1 = boundary[center_1]
    angle1 = math.pi  # 180度
    
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
    
    # 3. B'を中心にA'を180度回転してA''を得る
    angle2 = math.pi  # 180度
    dA_prime = A_prime - B_prime
    A_double_prime = np.array([
        math.cos(angle2) * dA_prime[0] - math.sin(angle2) * dA_prime[1],
        math.sin(angle2) * dA_prime[0] + math.cos(angle2) * dA_prime[1]
    ]) + B_prime
    
    # 4. A''からAへのベクトルがtrans_u
    trans_u = A_double_prime - A
    
    # trans_vの計算（上方向に読み替え）
    # 1. 辺1の最初の点Aと辺0の中点Bを取得
    A_v = boundary[edge_1[0]].copy()
    B_v = boundary[center_0].copy()
    
    # 2. 辺2の中点で180度回転させてA'とB'を得る
    center2 = boundary[center_2]
    
    # A_vを回転してA'を得る
    dA_v = A_v - center2
    A_prime_v = np.array([
        math.cos(angle1) * dA_v[0] - math.sin(angle1) * dA_v[1],
        math.sin(angle1) * dA_v[0] + math.cos(angle1) * dA_v[1]
    ]) + center2
    
    # B_vを回転してB'を得る
    dB_v = B_v - center2
    B_prime_v = np.array([
        math.cos(angle1) * dB_v[0] - math.sin(angle1) * dB_v[1],
        math.sin(angle1) * dB_v[0] + math.cos(angle1) * dB_v[1]
    ]) + center2
    
    # 3. B'を中心にA'を180度回転してA''を得る
    dA_prime_v = A_prime_v - B_prime_v
    A_double_prime_v = np.array([
        math.cos(angle2) * dA_prime_v[0] - math.sin(angle2) * dA_prime_v[1],
        math.sin(angle2) * dA_prime_v[0] + math.cos(angle2) * dA_prime_v[1]
    ]) + B_prime_v
    
    # 4. A''からAへのベクトルがtrans_v
    trans_v = A_double_prime_v - A_v
    
    return trans_u, trans_v


def number_of_vertices(m: int, n: int) -> int:
    return 2 * (m - 1) + 2 * (n - 1)


def edges(m, n):
    edge_0 = [i for i in range(m)]
    edge_1 = [i for i in range(m - 1, m - 1 + n)]
    edge_2 = [i for i in range(m + n - 2, 2 * m + n - 2)]
    edge_3 = [i for i in range(2 * m + n - 3, 2 * m + 2 * n - 3)]
    edge_3[-1] = 0  # 最後の頂点は最初の頂点と同じ
    return edge_0, edge_1, edge_2, edge_3

def center_of_edges(m, n):
    edge_0, edge_1, edge_2, edge_3 = edges(m, n)
    center_0 = edge_0[len(edge_0) // 2]
    center_1 = edge_1[len(edge_1) // 2]
    center_2 = edge_2[len(edge_2) // 2]
    center_3 = edge_3[len(edge_3) // 2]
    return center_0, center_1, center_2, center_3

def kernel(m: int, n: int):
    if m % 2 == 0 or n % 2 == 0:
        print("Error: p2のエッシャー化では、m,nは奇数である必要があります。")
        sys.exit(1)
    
    num_vertices = number_of_vertices(m, n)
    len_xy_vec = 2 * num_vertices
    # どのエッジがどのエッジに写るかで決まる。2はx,y成分の分
    # row_num = (
    #     (m - 1) * 2 #edge 0 が edge 2 に写る分
    #     + (n - 1) * 2 #edge 1 が edge 3 に写る分
    #     + (m - 1) * 2 #edge 2 が edge 0 に写る分
    #     + (n - 1) * 2 #edge 3 が edge 1 に写る分
    # )  
    # print(num_vertices, len_xy_vec)
    A = np.zeros((len_xy_vec, len_xy_vec))
    b = np.zeros((len_xy_vec,))

    edge_0, edge_1, edge_2, edge_3 = edges(m, n)
    center_0, center_1, center_2, center_3 = center_of_edges(m, n)
    
    # 各辺の中央インデックス（回転中心）
    c0_idx = len(edge_0) // 2
    c1_idx = len(edge_1) // 2
    c2_idx = len(edge_2) // 2
    c3_idx = len(edge_3) // 2
    
    theta = math.pi
    # 回転中心の頂点を除いて、前半と後半が180度回転で対応
    # edge_0[:c0_idx] が edge_0[c0_idx+1:][::-1] に対応
    tc.eq_rot(theta, edge_0[:c0_idx], edge_0[c0_idx+1:][::-1], center_0, A, b)
    tc.eq_rot(theta, edge_1[:c1_idx], edge_1[c1_idx+1:][::-1], center_1, A, b)
    tc.eq_rot(theta, edge_2[:c2_idx], edge_2[c2_idx+1:][::-1], center_2, A, b)
    tc.eq_rot(theta, edge_3[:c3_idx], edge_3[c3_idx+1:][::-1], center_3, A, b)

    # 最小二乗解（可解なら特殊解）
    x_p, residuals, rank, s = lstsq(A, b)

    # print("特殊解 x_p =", x_p)
    # print("A =", A)
    G = null_space(A)
    return (A, G, b, x_p)


def debug():
    kernel(5, 7)


def test():
    tran_u = (1, 0.2)
    tran_v = (0.1, 1)
    m = 7
    n = 7
    A, G, b, x_p = kernel(m, n)
    xx = []
    yy = []
    edge_0_x = np.linspace(0, tran_u[0], m)
    edge_0_y = np.linspace(0, tran_u[1], m) + np.sin(np.linspace(0, math.pi, m)) * 0.2
    edge_1_x = np.linspace(tran_u[0], tran_u[0] + tran_v[0], n)[1:] + np.sin(np.linspace(0, math.pi, n))[1:] * 0.1
    edge_1_y = np.linspace(tran_u[1], tran_u[1] + tran_v[1], n)[1:]
    edge_2_x = np.linspace(tran_u[0] + tran_v[0], tran_v[0], m)[1:]
    edge_2_y = np.linspace(tran_u[1] + tran_v[1], tran_v[1], m)[1:] + np.sin(np.linspace(0, math.pi, m))[1:] * 0.2
    edge_3_x = np.linspace(tran_v[0], 0, n)[1:-1] + np.sin(np.linspace(math.pi, 0, n)[1:-1]) * 0.1
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
    # # print(xy.shape, A.shape)
    # vec = A @ xy - b
    # print("A @ xy norm is expected to be close to 0:", np.linalg.norm(vec))

    w = xy 
    u_ = G @ (np.transpose(G) @ w)
    u = u_ + x_p
    redisual = A @ u - b
    print("A @ u - b norm is expected to be close to 0:", norm(redisual))
    xx = u[0::2]
    yy = u[1::2]
    
    # エッシャー化された境界を取得
    escherized_boundary = u.reshape(len(xx), 2)
    
    edge_0, edge_1, edge_2, edge_3 = edges(m, n)
    
    # エッシャー化後の各エッジの座標を取得
    u_edge_0 = np.array([[xx[i], yy[i]] for i in edge_0])
    u_edge_1 = np.array([[xx[i], yy[i]] for i in edge_1])
    u_edge_2 = np.array([[xx[i], yy[i]] for i in edge_2])
    u_edge_3 = np.array([[xx[i], yy[i]] for i in edge_3])
    
    theta = math.pi
    # 検証: rot(edge_0 at center_0) == reversed(edge_0)
    for i, (edge, u_edge) in enumerate([(edge_0, u_edge_0), (edge_1, u_edge_1), (edge_2, u_edge_2), (edge_3, u_edge_3)]):
        center = u_edge[len(edge) // 2]
        rot_edge = (
            np.array(
                [
                    [
                        math.cos(theta) * p[0] - math.sin(theta) * p[1],
                        math.sin(theta) * p[0] + math.cos(theta) * p[1],
                    ]
                    for p in u_edge - center
                ]
            )
            + center
        )
        reversed_edge = u_edge[::-1]
        error1 = np.linalg.norm(rot_edge - reversed_edge)
        print(f"\n検証: rot(edge_{i} at center) == reversed(edge_{i})")
        print(f"  誤差: {error1:.6e}")
    
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
    visualize_p2_matplotlib_tiling(escherized_boundary, m, n, ax[2], nstages=2)
    
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
        for j in range(200):
            image_rgb[i, j, 1] = (i + j) / 400  # 緑のグラデーション
    
    visualize_p2_pyvista_tiling(
        vertices,
        faces,
        uv_coords,
        image_rgb,
        escherized_boundary,
        m,
        n,
        "P2 Tiling with Texture",
        nx=2,
        ny=2,
    )


def visualize_p2_matplotlib_tiling(escherized_boundary, m, n, ax, nstages=2):
    """MatplotlibでP2タイリングを表示（境界線のみ）
    
    Args:
        escherized_boundary: エッシャー化された境界点 (N, 2)
        m: tran_u方向のエッジの頂点数
        n: tran_v方向のエッジの頂点数
        ax: matplotlib Axes object
        nstages: タイリングの段階数（デフォルト: 2）
    """
    # 中央の境界線
    bx = np.append(escherized_boundary[:, 0], escherized_boundary[0, 0])
    by = np.append(escherized_boundary[:, 1], escherized_boundary[0, 1])
    ax.plot(bx, by, "g-", linewidth=2)

    # P2の回転中心は各エッジの中心点（4点）
    center_0, center_1, center_2, center_3 = center_of_edges(m, n)
    rotation_centers = [
        (escherized_boundary[center_0, 0], escherized_boundary[center_0, 1]),
        (escherized_boundary[center_1, 0], escherized_boundary[center_1, 1]),
        (escherized_boundary[center_2, 0], escherized_boundary[center_2, 1]),
        (escherized_boundary[center_3, 0], escherized_boundary[center_3, 1]),
    ]
    
    # 回転中心に目印を表示
    for cx, cy in rotation_centers:
        ax.plot(cx, cy, 'ro', markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    # 回転角度（180度）
    r = math.pi
    
    # 生成されたタイルを記録（重複チェック用）
    generated_tiles = [escherized_boundary.copy()]
    
    # 第1段階: 各回転中心で180度回転
    for cx, cy in rotation_centers:
        dx = escherized_boundary[:, 0] - cx
        dy = escherized_boundary[:, 1] - cy
        rotated_x = np.cos(r) * dx - np.sin(r) * dy + cx
        rotated_y = np.sin(r) * dx + np.cos(r) * dy + cy
        
        # 重複チェック（中心座標で判定）
        rotated_boundary = np.column_stack([rotated_x, rotated_y])
        center_new = np.mean(rotated_boundary, axis=0)
        is_duplicate = False
        
        # for existing_tile in generated_tiles:
        #     existing_center = np.mean(existing_tile, axis=0)
        #     if np.linalg.norm(center_new - existing_center) < 1.0:
        #         is_duplicate = True
        #         break
        
        if not is_duplicate:
            # 閉曲線として描画
            rotated_x_closed = np.append(rotated_x, rotated_x[0])
            rotated_y_closed = np.append(rotated_y, rotated_y[0])
            ax.plot(rotated_x_closed, rotated_y_closed, "g-", linewidth=2, alpha=0.7)
            generated_tiles.append(rotated_boundary)
    
    # 多段階タイリング
    for stage in range(2, nstages + 1):
        new_tiles = []
        for tile in generated_tiles[1:]:  # 中央タイル以外
            # このタイル自身の辺の中心点を回転中心として使用
            tile_rotation_centers = [
                (tile[center_0, 0], tile[center_0, 1]),
                (tile[center_1, 0], tile[center_1, 1]),
                (tile[center_2, 0], tile[center_2, 1]),
                (tile[center_3, 0], tile[center_3, 1]),
            ]
            
            # このタイルの回転中心に目印を表示
            for cx, cy in tile_rotation_centers:
                ax.plot(cx, cy, 'bo', markersize=6, markeredgecolor='black', markeredgewidth=0.5, alpha=0.6)
            
            for cx, cy in tile_rotation_centers:
                dx = tile[:, 0] - cx
                dy = tile[:, 1] - cy
                rotated_x = np.cos(r) * dx - np.sin(r) * dy + cx
                rotated_y = np.sin(r) * dx + np.cos(r) * dy + cy
                
                # 重複チェック
                rotated_boundary = np.column_stack([rotated_x, rotated_y])
                center_new = np.mean(rotated_boundary, axis=0)
                is_duplicate = False
                
                for existing_tile in generated_tiles + new_tiles:
                    existing_center = np.mean(existing_tile, axis=0)
                    if np.linalg.norm(center_new - existing_center) < 1.0:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    rotated_x_closed = np.append(rotated_x, rotated_x[0])
                    rotated_y_closed = np.append(rotated_y, rotated_y[0])
                    ax.plot(rotated_x_closed, rotated_y_closed, "g-", linewidth=2, alpha=0.7)
                    new_tiles.append(rotated_boundary)
        
        generated_tiles.extend(new_tiles)
    
    ax.set_title(f"P2 Tiling - Boundary Only ({len(generated_tiles)} tiles)")
    ax.set_aspect("equal")


def visualize_p2_pyvista_tiling(
    deformed_vertices,
    faces,
    uv_coords,
    image_rgb,
    escherized_boundary,
    m,
    n,
    title="P2 Tiling with Texture",
    nx=2,
    ny=2,
):
    """PyVistaでP2タイリングされたテクスチャ付きメッシュを表示
    
    Args:
        deformed_vertices: 変形後の頂点座標 (V, 2)
        faces: 三角形インデックス (F, 3)
        uv_coords: UV座標 (V, 2)
        image_rgb: RGB画像 (H, W, 3), 0-1の範囲
        escherized_boundary: エッシャー化された境界点 (N, 2)
        m: tran_u方向のエッジの頂点数
        n: tran_v方向のエッジの頂点数
        title: 表示タイトル
        nx: x方向のタイル数（基本パターン単位）
        ny: y方向のタイル数（基本パターン単位）
    """
    print(f"P2タイリング表示: {len(deformed_vertices)}頂点, {len(faces)}面")

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

    # P2対称性の回転中心点（各エッジの中心点）
    center_0, center_1, center_2, center_3 = center_of_edges(m, n)
    rotation_centers = [
        (escherized_boundary[center_0, 0], escherized_boundary[center_0, 1]),
        (escherized_boundary[center_1, 0], escherized_boundary[center_1, 1]),
        (escherized_boundary[center_2, 0], escherized_boundary[center_2, 1]),
        (escherized_boundary[center_3, 0], escherized_boundary[center_3, 1]),
    ]

    # PyVista用のメッシュ作成のヘルパー関数
    def create_mesh(vertices_2d):
        vertices_3d = np.column_stack([vertices_2d, np.zeros(len(vertices_2d))])
        faces_pv = np.column_stack([np.full(len(faces), 3), faces]).flatten()
        mesh = pv.PolyData(vertices_3d, faces_pv)
        mesh.active_texture_coordinates = uv_coords
        return mesh

    # 頂点を指定の中心で回転させるヘルパー関数
    def rotate_vertices(vertices_2d, center, angle):
        cx, cy = center
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx, dy = vertices_2d[:, 0] - cx, vertices_2d[:, 1] - cy
        rotated_x = cos_a * dx - sin_a * dy + cx
        rotated_y = sin_a * dx + cos_a * dy + cy
        return np.column_stack([rotated_x, rotated_y])

    # P2パターン: 基本領域（4つのタイル）を作成
    base_pattern_tiles = []  # 基本パターンのタイル頂点リスト
    
    # 中央タイル
    base_pattern_tiles.append(deformed_vertices.copy())
    
    angle = math.pi  # 180度
    
    # 辺1で180度回転したタイル（タイルA）
    center_1_coord = rotation_centers[1]  # 辺1の中心
    tile_A = rotate_vertices(deformed_vertices, center_1_coord, angle)
    base_pattern_tiles.append(tile_A)
    
    # 辺2で180度回転したタイル（タイルB）
    center_2_coord = rotation_centers[2]  # 辺2の中心
    tile_B = rotate_vertices(deformed_vertices, center_2_coord, angle)
    base_pattern_tiles.append(tile_B)
    
    # タイルAの辺0の中心で180度回転（タイルC）
    # タイルAのエッジを計算する必要がある
    edge_0, edge_1, edge_2, edge_3 = edges(m, n)
    center_0_idx = center_0  # 辺0の中心インデックス
    # タイルAの辺0の中心座標を取得
    tile_A_center_0 = (tile_A[center_0_idx, 0], tile_A[center_0_idx, 1])
    tile_C = rotate_vertices(tile_A, tile_A_center_0, angle)
    base_pattern_tiles.append(tile_C)
    
    # 基本パターンの並進ベクトルを計算
    tran_vec_x, tran_vec_y = trans_uv(escherized_boundary, m, n)
    
    print(f"    並進ベクトル u: ({tran_vec_x[0]:.4f}, {tran_vec_x[1]:.4f}), 長さ: {np.linalg.norm(tran_vec_x):.4f}")
    print(f"    並進ベクトル v: ({tran_vec_y[0]:.4f}, {tran_vec_y[1]:.4f}), 長さ: {np.linalg.norm(tran_vec_y):.4f}")
    
    tile_count = 0
    
    # nx × ny の範囲でタイルを配置
    for i in range(nx//2):
        for j in range(ny//2):
            translation = i * tran_vec_x + j * tran_vec_y
            
            # 基本パターンの各タイルを並進
            for tile_verts in base_pattern_tiles:
                translated_verts = tile_verts + translation
                
                # メッシュを追加
                mesh = create_mesh(translated_verts)
                plotter.add_mesh(mesh, texture=texture, show_edges=False, opacity=0.9)
                
                tile_count += 1
    
    print(f"  P2タイリング完了: {tile_count}個のタイル")

    # カメラを2D表示用に設定
    plotter.view_xy()
    plotter.camera.parallel_projection = True
    plotter.camera.zoom(0.5)

    plotter.add_title(f"{title}\n{tile_count} tiles total", font_size=12)
    plotter.set_background("white")

    print(f"  P2タイリング完了: {tile_count}個のタイル")
    print(
        "  表示オプション: マウスドラッグ=回転, Shift+ドラッグ=パン, ホイール=ズーム, q=終了"
    )

    plotter.show()


if __name__ == "__main__":
    from matplotlib import pyplot
    import pyvista as pv
    # debug()
    test()
