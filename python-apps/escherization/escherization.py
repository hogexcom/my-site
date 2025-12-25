#!/usr/bin/env python3
"""
P3 Escherization with Bilaplacian Deformation

画像から輪郭を抽出し、P3対称性を持つエッシャー風タイリングを生成
重ラプラシアンデフォームによる内部頂点の滑らかな変形を適用

機能:
1. 画像から輪郭抽出
2. 6m点への等弧長リサンプリング
3. P3対称性に最適な開始点探索
4. 三角形メッシュ生成
5. 重ラプラシアンデフォーム
6. PyVistaによるテクスチャ付きタイリング表示
"""

import numpy as np
from matplotlib import pyplot
import matplotlib.tri as mtri
import os
import sys
import argparse
import math
import cv2
import triangle as tr
import pyvista as pv
import igl
import urllib.request
import tempfile

import p3
import p1
import p2
from escherize_boundary import escherization
from contour_extract import prepare_contour_for_escherization
from triangulate_mesh import triangulate_boundary
from mesh_deformation import solve_bilaplacian_deformation, solve_arap_deformation
from visualize_matplotlib import visualize_escherization_matplotlib
from visualize_pyvista import visualize_escherization_pyvista


def compute_principal_directions(contour: np.ndarray) -> tuple:
    """
    最小外接矩形から2辺のベクトルを計算（輪郭面積と一致するように調整）

    Args:
        contour: 閉曲線の点列 (n, 2)

    Returns:
        v1: 第1辺のベクトル (2,) - 長辺方向
        v2: 第2辺のベクトル (2,) - 短辺方向

    Note:
        - OpenCVのminAreaRectで最小外接矩形を計算
        - 矩形の面積と輪郭の面積が一致するように2辺を拡大縮小
        - 2辺は直交する

    Example:
        >>> contour = np.array([[0, 0], [2, 0], [2, 1], [0, 1]])  # 横長の矩形
        >>> v1, v2 = compute_principal_directions(contour)
        >>> # v1 は長辺ベクトル, v2 は短辺ベクトル
    """
    # OpenCVのminAreaRectはfloat32配列を要求
    contour_cv = contour.astype(np.float32)

    # 最小外接矩形を計算
    # rect = ((center_x, center_y), (width, height), angle)
    rect = cv2.minAreaRect(contour_cv)
    center, size, angle_deg = rect

    # 矩形のサイズと角度
    width, height = size
    angle_rad = np.radians(angle_deg)

    # 矩形の面積
    rect_area = width * height

    # 輪郭の面積を計算（Shoelace formula）
    n = len(contour)
    contour_area = 0.0
    for i in range(n):
        j = (i + 1) % n
        contour_area += contour[i, 0] * contour[j, 1]
        contour_area -= contour[j, 0] * contour[i, 1]
    contour_area = abs(contour_area) / 2.0

    # 面積比を計算して拡大縮小係数を求める
    scale_factor = np.sqrt(contour_area / rect_area)

    # 調整後のサイズ
    adjusted_width = width * scale_factor
    adjusted_height = height * scale_factor

    # 矩形の2辺の方向ベクトルを計算
    # OpenCVの角度定義: width方向の角度
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # 第1辺（width方向）のベクトル
    v1 = np.array([adjusted_width * cos_a, adjusted_width * sin_a])

    # 第2辺（height方向、90度回転）のベクトル
    v2 = np.array([-adjusted_height * sin_a, adjusted_height * cos_a])

    # 検証: 調整後の矩形面積
    adjusted_rect_area = abs(np.cross(v1, v2))

    print(f"最小外接矩形から2辺ベクトルを計算:")
    print(f"  元の矩形サイズ: {width:.2f} × {height:.2f}")
    print(f"  元の矩形面積: {rect_area:.2f}")
    print(f"  輪郭の面積: {contour_area:.2f}")
    print(f"  拡大縮小係数: {scale_factor:.4f}")
    print(f"  調整後サイズ: {adjusted_width:.2f} × {adjusted_height:.2f}")
    print(f"  調整後面積: {adjusted_rect_area:.2f}")
    print(f"  v1 = ({v1[0]:.2f}, {v1[1]:.2f}), |v1| = {np.linalg.norm(v1):.2f}")
    print(f"  v2 = ({v2[0]:.2f}, {v2[1]:.2f}), |v2| = {np.linalg.norm(v2):.2f}")
    print(
        f"  アスペクト比: {max(adjusted_width, adjusted_height) / min(adjusted_width, adjusted_height):.2f}:1"
    )

    return v1, v2


def rotate_array(arr: np.ndarray, start_idx: int) -> np.ndarray:
    """配列を回転して start_idx を先頭にする"""
    return np.roll(arr, -start_idx, axis=0)


def download_image_from_url(url: str) -> str:
    """
    URLから画像をダウンロードして一時ファイルとして保存
    
    Args:
        url: 画像URL
    
    Returns:
        一時ファイルのパス
    """
    print(f"URLから画像をダウンロード中: {url}")
    try:
        # 一時ファイルを作成（拡張子を推定）
        suffix = os.path.splitext(url)[1] or '.png'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = temp_file.name
        temp_file.close()
        
        # URLからダウンロード
        urllib.request.urlretrieve(url, temp_path)
        print(f"ダウンロード完了: {temp_path}")
        return temp_path
    except Exception as e:
        print(f"エラー: URLからのダウンロードに失敗しました: {e}")
        raise


def main(image_path: str = None):
    # 一時ファイルのパス（削除用）
    temp_file_path = None
    
    try:
        if image_path is None:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(dir_path)  # プロジェクトルート（mathworks）
            image_path = os.path.join(project_root,  "yumekawa_animal_usagi.png")
        elif image_path.startswith(('http://', 'https://')):
            # URLからダウンロード
            temp_file_path = download_image_from_url(image_path)
            image_path = temp_file_path

        # ========== パラメータ設定 ==========
        # 境界抽出パラメータ
        simplify_ratio = 0.005  # 輪郭簡略化率（小さいほど細かい: 0.001-0.02）

        # 境界調整方法
        use_resample = True  # True: 等間隔再サンプリング, False: 最近傍補間

        # 変形方法
        deformation_method = "arap"  # "bilaplacian" or "arap"

        # タイリング段階数
        max_stages = 4  # 最大段階数（デフォルト4）

        # タイリングパターン
        tiling_pattern = "P2"  # "P3, P1" "P2" のみ対応

        m = 21  # P3の1辺の頂点数
        n = 21  # P1,P2の1辺の頂点数
        # ====================================

        print("=" * 60)
        print("P3 Escherization with Bilaplacian Deformation")
        print("=" * 60)

        # 1. 画像から輪郭抽出、反時計回り統一、平行四辺形ベクトル計算（統合関数）
        X, image, tran_u, tran_v = prepare_contour_for_escherization(
            image_path,
            simplify_ratio=simplify_ratio,
            theta1=0,
            theta2=np.pi / 2,
            verbose=True,
        )

        # 6. 最適な開始点を探索
        escherized_boundary, X_best, best_idx, best_error, errors = escherization(
            X, tiling_pattern, m, n, use_resample, tran_u, tran_v
        )
        print(f"\n=== 開始点探索結果 ===")
        print(f"最適開始点: {best_idx}")
        print(f"最小誤差: {best_error:.2f}")

        # 8. メッシュを作成
        print("\n=== メッシュ作成 ===")
        vertices, faces, boundary_indices = triangulate_boundary(X_best)
        print(f"頂点数: {len(vertices)}, 三角形数: {len(faces)}")

        # 10. メッシュ変形（BilaplacianまたはARAP）
        if deformation_method == "bilaplacian":
            print("\n=== 重ラプラシアンデフォーム ===")
            deformed_vertices = solve_bilaplacian_deformation(
                    vertices_original=vertices,
                faces=faces,
                boundary_indices=boundary_indices,
                boundary_positions=escherized_boundary,
                constraint_weight=1000.0,  # 境界条件を強く適用
            )
            print("デフォーム完了")
        else:
            print("\n=== ARAPデフォーム ===")
            deformed_vertices = solve_arap_deformation(
                vertices_original=vertices,
                faces=faces,
                boundary_indices=boundary_indices,
                boundary_positions=escherized_boundary,
            )
            print("デフォーム完了")

        # デバッグ: 境界点の一致を確認
        boundary_error = np.max(
            np.linalg.norm(
                deformed_vertices[boundary_indices] - escherized_boundary, axis=1
            )
        )
        print(f"境界点の最大誤差: {boundary_error:.2e}")
        if boundary_error > 1e-3:
            print(
                f"警告: 変形後の境界点がエッシャー化境界と一致していません（誤差: {boundary_error:.3f}）"
            )

        # ===== Matplotlib表示（簡略版：線画のみ） =====
        visualize_escherization_matplotlib(
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

        # ===== PyVista表示（テクスチャ付き） =====
        visualize_escherization_pyvista(
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
            nx=10,
            ny=10,
            tran_u=tran_u,
            tran_v=tran_v,
        )
    
    finally:
        # 一時ファイルを削除
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"\n一時ファイルを削除しました: {temp_file_path}")
            except Exception as e:
                print(f"\n警告: 一時ファイルの削除に失敗しました: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P3 Escherization with Bilaplacian Deformation"
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Path to input image or URL (default: data/yumekawa_animal_usagi.png)"
    )

    args = parser.parse_args()
    main(args.image_path)
