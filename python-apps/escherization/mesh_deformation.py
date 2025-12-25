"""
メッシュ変形モジュール

2つの変形手法を提供:
1. Bilaplacian Deformation (重ラプラシアン変形) - 滑らかな変形
2. ARAP Deformation (As-Rigid-As-Possible) - 剛体に近い変形

使用方法:
    python mesh_deformation.py image.png --method arap
    python mesh_deformation.py image.png --method bilaplacian
"""

import numpy as np
import cv2
import argparse
from scipy import sparse
from scipy.sparse.linalg import spsolve
import igl
from triangulate_mesh import extract_contour_simple, triangulate_boundary


def compute_cotangent_laplacian(
    vertices: np.ndarray, faces: np.ndarray
) -> sparse.csr_matrix:
    """
    余接ラプラシアン行列を計算

    L[i,j] = -w_ij (i != j, 隣接)
    L[i,i] = Σ w_ij

    w_ij = (cot α_ij + cot β_ij) / 2

    Args:
        vertices: 頂点座標 (V, 2)
        faces: 三角形インデックス (F, 3)

    Returns:
        L: ラプラシアン行列 (V, V)
    """
    n = len(vertices)

    # COO形式で構築
    rows = []
    cols = []
    data = []

    for face in faces:
        for k in range(3):
            i = face[k]
            j = face[(k + 1) % 3]
            l = face[(k + 2) % 3]

            # 頂点位置
            vi = vertices[i]
            vj = vertices[j]
            vl = vertices[l]

            # lから見たiとjへのベクトル
            e1 = vi - vl
            e2 = vj - vl

            # cotangent = cos/sin = dot/(cross magnitude)
            dot = np.dot(e1, e2)
            cross = abs(e1[0] * e2[1] - e1[1] * e2[0])

            if cross > 1e-10:
                cot = dot / cross
            else:
                cot = 0.0

            weight = cot / 2.0

            # L[i,j] と L[j,i] に -weight を加算
            rows.extend([i, j, i, j])
            cols.extend([j, i, i, j])
            data.extend([-weight, -weight, weight, weight])

    L = sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
    return L.tocsr()


def solve_bilaplacian_deformation(
    vertices_original: np.ndarray,
    faces: np.ndarray,
    boundary_indices: np.ndarray,
    boundary_positions: np.ndarray,
    constraint_weight: float = 1000.0,
) -> np.ndarray:
    """
    重ラプラシアン変形を解く

    境界頂点を固定し、内部頂点を最適化
    最小化: ||L² v||² subject to v[boundary] = boundary_positions

    Args:
        vertices_original: 元の頂点座標 (V, 2)
        faces: 三角形インデックス (F, 3)
        boundary_indices: 境界頂点のインデックス (B,)
        boundary_positions: 境界頂点の新しい位置 (B, 2)
        constraint_weight: 境界条件の重み（大きいほど強い制約）

    Returns:
        deformed_vertices: 変形後の頂点座標 (V, 2)
    """
    n = len(vertices_original)

    # ラプラシアン行列を計算
    L = compute_cotangent_laplacian(vertices_original, faces)

    # 重ラプラシアン行列 L²
    L2 = L @ L

    # エネルギー項: E = v^T L² L² v (2次形式)
    # ∇E = 2 L² L² v = 0 → L² L² v = 0
    # しかし境界条件があるので、ペナルティ法を使用

    # システム行列: A = L² L² + λ C^T C
    # C: 境界条件行列（境界頂点に対応する行のみ1）
    A = L2.T @ L2

    # 境界条件をペナルティ法で追加
    # 境界頂点の行に大きな対角要素を追加
    for idx in boundary_indices:
        A[idx, idx] += constraint_weight

    # 右辺ベクトル（x座標とy座標を別々に解く）
    deformed_vertices = np.zeros_like(vertices_original)

    for dim in range(2):  # x, y
        b = np.zeros(n)

        # 境界条件を右辺に設定
        for i, idx in enumerate(boundary_indices):
            b[idx] = constraint_weight * boundary_positions[i, dim]

        # 連立方程式を解く
        deformed_vertices[:, dim] = spsolve(A, b)

    return deformed_vertices


def solve_arap_deformation(
    vertices_original: np.ndarray,
    faces: np.ndarray,
    boundary_indices: np.ndarray,
    boundary_positions: np.ndarray,
) -> np.ndarray:
    """
    ARAP変形 (As-Rigid-As-Possible Deformation)

    libiglを使用して、各三角形が可能な限り剛体変換に近くなるように変形

    Args:
        vertices_original: 元の頂点座標 (V, 2)
        faces: 三角形インデックス (F, 3)
        boundary_indices: 境界頂点のインデックス (B,)
        boundary_positions: 境界頂点の新しい位置 (B, 2)

    Returns:
        deformed_vertices: 変形後の頂点座標 (V, 2)
    """
    # 2D頂点を3Dに拡張（libiglは3D座標を要求）
    vertices_3d = np.column_stack([vertices_original, np.zeros(len(vertices_original))])
    boundary_positions_3d = np.column_stack(
        [boundary_positions, np.zeros(len(boundary_positions))]
    )

    # 境界頂点を固定点として設定
    fixed_indices = boundary_indices.astype(np.int32)

    # ARAPデータ構造を作成
    arap_data = igl.ARAPData()

    # 前処理
    igl.arap_precomputation(
        vertices_3d.astype(np.float64),
        faces.astype(np.int64),
        3,  # 次元
        fixed_indices,
        arap_data,
    )

    # ARAP解を計算（反復法）
    vertices_3d_deformed = igl.arap_solve(
        boundary_positions_3d.astype(np.float64),
        arap_data,
        vertices_3d.astype(np.float64),
    )

    # 2D座標に戻す
    deformed_vertices = vertices_3d_deformed[:, :2].copy()

    return deformed_vertices


def test_mesh_deformation(
    image_path: str,
    method: str = "arap",
    simplify_ratio: float = 0.008,
    deform_scale: float = 0.3,
):
    """
    メッシュ変形のテスト

    輪郭を抽出してメッシュを生成し、境界を変形してから内部を変形

    Args:
        image_path: 画像ファイルパス
        method: 変形手法 ("arap" or "bilaplacian")
        simplify_ratio: 輪郭簡略化の比率
        deform_scale: 変形の大きさ（0-1、境界をどれだけ動かすか）
    """
    import matplotlib.pyplot as plt

    print("=" * 60)
    print(f"Testing Mesh Deformation: {method.upper()}")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Simplify ratio: {simplify_ratio}")
    print(f"Deformation scale: {deform_scale}")

    # 1. 輪郭抽出
    contour, image = extract_contour_simple(image_path, simplify_ratio)
    print(f"\nContour: {len(contour)} points")

    # 2. メッシュ生成
    vertices, faces, boundary_indices = triangulate_boundary(contour)
    print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")

    # 3. 境界を変形（例: 中心に向かって縮小）
    boundary_verts = vertices[boundary_indices]
    center = np.mean(boundary_verts, axis=0)

    # 境界を中心に向かって移動
    deformed_boundary = boundary_verts + (center - boundary_verts) * deform_scale

    print(f"\n=== {method.upper()} Deformation ===")

    # 4. メッシュ変形
    if method == "bilaplacian":
        print("Computing bilaplacian deformation...")
        deformed_vertices = solve_bilaplacian_deformation(
            vertices,
            faces,
            boundary_indices,
            deformed_boundary,
            constraint_weight=1000.0,
        )
    elif method == "arap":
        print("Computing ARAP deformation...")
        print("  ARAP precomputation...")
        deformed_vertices = solve_arap_deformation(
            vertices, faces, boundary_indices, deformed_boundary
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    print("Deformation complete")

    # 5. 境界誤差を確認
    boundary_error = np.max(
        np.linalg.norm(deformed_vertices[boundary_indices] - deformed_boundary, axis=1)
    )
    print(f"\nBoundary constraint error: {boundary_error:.2e}")

    # 6. 変形の統計
    displacements = np.linalg.norm(deformed_vertices - vertices, axis=1)
    print(f"\nDisplacement statistics:")
    print(f"  Min: {np.min(displacements):.2f}")
    print(f"  Max: {np.max(displacements):.2f}")
    print(f"  Mean: {np.mean(displacements):.2f}")
    print(f"  Std: {np.std(displacements):.2f}")

    # 7. 可視化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 左: 元のメッシュ
    ax1 = axes[0]
    ax1.set_aspect("equal")
    ax1.triplot(vertices[:, 0], vertices[:, 1], faces, "b-", linewidth=0.5, alpha=0.5)
    ax1.plot(
        boundary_verts[:, 0], boundary_verts[:, 1], "ro", markersize=4, label="Boundary"
    )
    ax1.invert_yaxis()
    ax1.set_title("Original Mesh")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 中央: 変形後のメッシュ
    ax2 = axes[1]
    ax2.set_aspect("equal")
    ax2.triplot(
        deformed_vertices[:, 0],
        deformed_vertices[:, 1],
        faces,
        "r-",
        linewidth=0.5,
        alpha=0.5,
    )
    ax2.plot(
        deformed_boundary[:, 0],
        deformed_boundary[:, 1],
        "go",
        markersize=4,
        label="Deformed Boundary",
    )
    ax2.invert_yaxis()
    ax2.set_title(f"Deformed Mesh ({method.upper()})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 右: 変形量の可視化
    ax3 = axes[2]
    ax3.set_aspect("equal")

    # 変形量で色分け
    triangulation = plt.matplotlib.tri.Triangulation(
        deformed_vertices[:, 0], deformed_vertices[:, 1], faces
    )

    # 各三角形の変形量（頂点の平均）
    tri_displacements = []
    for face in faces:
        tri_disp = np.mean(displacements[face])
        tri_displacements.append(tri_disp)

    tcf = ax3.tripcolor(
        triangulation, tri_displacements, cmap="viridis", shading="flat"
    )
    ax3.triplot(
        deformed_vertices[:, 0],
        deformed_vertices[:, 1],
        faces,
        "k-",
        linewidth=0.3,
        alpha=0.3,
    )
    plt.colorbar(tcf, ax=ax3, label="Displacement")

    ax3.invert_yaxis()
    ax3.set_title("Displacement Magnitude")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)


def main():
    """
    メイン関数: コマンドライン引数で画像と変形手法を指定してテストを実行
    """
    parser = argparse.ArgumentParser(
        description="Test mesh deformation methods (ARAP and Bilaplacian)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test ARAP deformation (default)
  python mesh_deformation.py rabbit.png
  
  # Test Bilaplacian deformation
  python mesh_deformation.py rabbit.png --method bilaplacian
  
  # Custom parameters
  python mesh_deformation.py rabbit.png --method arap --simplify 0.005 --deform 0.5
  
  # Compare both methods
  python mesh_deformation.py rabbit.png --method arap
  python mesh_deformation.py rabbit.png --method bilaplacian
        """,
    )

    parser.add_argument("image", help="Input image file path")
    parser.add_argument(
        "--method",
        choices=["arap", "bilaplacian"],
        default="arap",
        help="Deformation method (default: arap)",
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=0.008,
        help="Contour simplification ratio (default: 0.008)",
    )
    parser.add_argument(
        "--deform", type=float, default=0.3, help="Deformation scale 0-1 (default: 0.3)"
    )

    args = parser.parse_args()

    # テスト実行
    test_mesh_deformation(args.image, args.method, args.simplify, args.deform)


if __name__ == "__main__":
    main()
