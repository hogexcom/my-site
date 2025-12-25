#!/usr/bin/env python3
"""
PyVista Visualization for Escherization

エッシャー化プロセスのPyVistaによる3D可視化モジュール
テクスチャ付きタイリングの3D表示を提供
"""

import numpy as np
import cv2
import pyvista as pv
import p3
import p1
import p2


def compute_uv_coordinates(
    mesh_vertices, boundary_indices, original_boundary, img_shape
):
    """
    改善されたUV座標計算：画像全体を適切に使用

    Args:
        mesh_vertices: メッシュ頂点座標 (nv, 2)
        boundary_indices: 境界頂点のインデックス配列
        original_boundary: 元の境界座標 (n, 2)
        img_shape: 画像の形状 (height, width, channels)

    Returns:
        uv: UV座標配列 (nv, 2) - 各頂点の2Dテクスチャ座標 [0, 1]

    Note:
        - 境界頂点は元の画像座標から直接UV座標を計算
        - 内部頂点はメッシュと画像の境界ボックス対応から補間
    """
    h, w = img_shape[:2]
    uv = np.zeros((len(mesh_vertices), 2))

    # 境界頂点のUVを設定（正規化）
    for i, boundary_idx in enumerate(boundary_indices):
        if i < len(original_boundary):
            orig_point = original_boundary[i]
            uv[boundary_idx, 0] = orig_point[0] / w  # 0-1の範囲に正規化
            uv[boundary_idx, 1] = 1.0 - orig_point[1] / h  # Y軸反転して正規化

    # 内部頂点のUV座標を適切に補間
    # メッシュの境界ボックスと画像の境界ボックスの対応を計算
    boundary_verts = mesh_vertices[boundary_indices]
    mesh_bbox_min = np.min(boundary_verts, axis=0)
    mesh_bbox_max = np.max(boundary_verts, axis=0)
    mesh_bbox_size = mesh_bbox_max - mesh_bbox_min

    # 画像座標での境界ボックス
    img_bbox_min = np.min(original_boundary, axis=0)
    img_bbox_max = np.max(original_boundary, axis=0)
    img_bbox_size = img_bbox_max - img_bbox_min

    # 内部頂点のUV座標を計算
    for i in range(len(mesh_vertices)):
        if i not in boundary_indices:
            # メッシュ座標での相対位置
            rel_pos = (mesh_vertices[i] - mesh_bbox_min) / (mesh_bbox_size + 1e-8)

            # 画像座標に変換
            img_coord = img_bbox_min + rel_pos * img_bbox_size

            # UV座標に正規化
            uv[i, 0] = img_coord[0] / w
            uv[i, 1] = 1.0 - img_coord[1] / h

            # UV座標をクランプ
            uv[i] = np.clip(uv[i], 0.0, 1.0)

    return uv


def visualize_escherization_pyvista(
    X_best,
    X,
    vertices,
    faces,
    deformed_vertices,
    boundary_indices,
    image,
    escherized_boundary,
    tiling_pattern,
    m,
    n,
    nx=4,
    ny=4,
    tran_u=None,
    tran_v=None,
):
    """
    エッシャー化プロセスのPyVista可視化（テクスチャ付き）

    Args:
        X_best: 最適開始点で回転した元の境界 (n, 2)
        X: 元の境界（回転前） (n, 2)
        vertices: メッシュ頂点（元の位置） (nv, 2)
        faces: メッシュ面 (nf, 3)
        deformed_vertices: 変形後のメッシュ頂点 (nv, 2)
        boundary_indices: 境界頂点のインデックス配列
        image: 元画像（BGR/BGRA/Grayscale）
        escherized_boundary: エッシャー化境界 (n, 2)
        tiling_pattern: タイリングパターン ("P3", "P1", "P2")
        m: P3/P2の1辺の頂点数
        n: P1/P2の1辺の頂点数
        max_stages: タイリング段階数（P3のみ）
        tran_u: P1タイリング用の平行移動ベクトル1 (2,)
        tran_v: P1タイリング用の平行移動ベクトル2 (2,)
        show_boundary: 境界線を表示するかどうか (デフォルト: True)

    Note:
        - PyVistaで3Dテクスチャ付きタイリングを表示
        - 画像をRGBに変換し、UV座標でマッピング
        - show_boundaryがTrueの場合、境界線を黒線で表示
    """
    print("\n=== PyVista表示 ===")

    # 画像をRGB変換
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb / 255.0

    # 元の境界との対応を取る（重心移動は行っていないので直接使用）
    original_boundary_matched = np.zeros_like(X_best)
    for i in range(len(X_best)):
        distances = np.sum((X - X_best[i]) ** 2, axis=1)
        closest_idx = np.argmin(distances)
        original_boundary_matched[i] = X[closest_idx]

    try:
        # UV座標計算
        original_uv = compute_uv_coordinates(
            vertices, boundary_indices, original_boundary_matched, image.shape
        )
        deformed_uv = compute_uv_coordinates(
            deformed_vertices, boundary_indices, original_boundary_matched, image.shape
        )

        # タイリング（テクスチャ付き）
        if tiling_pattern == "P3":
            p3.visualize_p3_pyvista_tiling(
                deformed_vertices,
                faces,
                deformed_uv,
                image_rgb,
                escherized_boundary,
                m,
                "P3 Tiling with Texture",
                nx=nx,
                ny=ny,
            )
        elif tiling_pattern == "P1":
            if tran_u is None or tran_v is None:
                raise ValueError("P1タイリングにはtran_u, tran_vが必要です")
            p1.visualize_p1_pyvista_tiling(
                deformed_vertices,
                faces,
                deformed_uv,
                image_rgb,
                tran_u,
                tran_v,
                "P1 Tiling with Texture",
                nx=nx,
                ny=ny,
            )
        elif tiling_pattern == "P2":
            p2.visualize_p2_pyvista_tiling(
                deformed_vertices,
                faces,
                deformed_uv,
                image_rgb,
                escherized_boundary,
                m,
                n,
                "P2 Tiling with Texture",
                nx=nx,
                ny=ny,
            )

    except Exception as e:
        print(f"PyVista表示エラー: {e}")
        import traceback

        traceback.print_exc()


def test_visualize_pyvista():
    """
    PyVista可視化のテスト関数

    歪んだ六角形境界でP3タイリングをテスト
    """
    import os
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

    X = np.column_stack([xx, yy])

    # エッシャー化アルゴリズムを実際に適用
    from escherize_boundary import escherization

    escherized_boundary, X_rotated, best_idx, best_error, errors = escherization(
        X, "P3", m, m, resample=True
    )

    # メッシュ生成
    from triangulate_mesh import triangulate_boundary

    vertices, faces, boundary_indices = triangulate_boundary(X_rotated)

    # ARAP変形でエッシャー化境界に合わせる
    from mesh_deformation import solve_arap_deformation

    deformed_vertices = solve_arap_deformation(
        vertices, faces, boundary_indices, escherized_boundary
    )

    # ダミー画像作成
    dir_path = os.path.dirname(os.path.abspath(__file__))
    image_path = dir_path + "/../yumekawa_animal_usagi.png"

    if os.path.exists(image_path):
        image = cv2.imread(image_path)
    else:
        # ダミー画像
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128

    # 可視化
    visualize_escherization_pyvista(
        X_rotated,
        X,
        vertices,
        faces,
        deformed_vertices,
        boundary_indices,
        image,
        escherized_boundary,
        tiling_pattern="P3",
        m=m,
        n=21,
        nx=4,
        ny=4,
    )


if __name__ == "__main__":
    test_visualize_pyvista()
