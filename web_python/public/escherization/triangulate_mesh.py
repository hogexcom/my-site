"""
境界点から三角形メッシュを生成するモジュール

使用方法:
    python triangulate_mesh.py image.png
    python triangulate_mesh.py image.png --simplify 0.005
    python triangulate_mesh.py image.png --max-area 100
"""

import numpy as np
import cv2
import argparse
import triangle as tr


def extract_contour_simple(image_path: str, simplify_ratio: float = 0.008) -> tuple:
    """画像から最大の輪郭を抽出（簡易版）

    Args:
        image_path: 画像ファイルパス
        simplify_ratio: 輪郭簡略化の比率（小さいほど細かい）

    Returns:
        contour: 輪郭点列 (n, 2)
        image: 元画像
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"画像を読み込めません: {image_path}")

    # アルファチャンネルで二値化
    if len(image.shape) == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        _, binary = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # ノイズ除去
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 輪郭抽出
    contours_raw, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours_raw:
        raise ValueError("輪郭が見つかりません")

    # 最大面積の輪郭を選択
    max_contour = max(contours_raw, key=cv2.contourArea)

    # Douglas-Peucker簡略化
    perimeter = cv2.arcLength(max_contour, True)
    epsilon = simplify_ratio * perimeter
    simplified = cv2.approxPolyDP(max_contour, epsilon, True)

    points = simplified.reshape(-1, 2).astype(np.float64)

    return points, image


def triangulate_boundary(boundary: np.ndarray, max_area: float = None) -> tuple:
    """
    境界点からメッシュを作成

    Args:
        boundary: 境界点の配列 (N, 2)
        max_area: 三角形の最大面積（Noneの場合は自動計算）

    Returns:
        vertices: 全頂点 (V, 2)
        faces: 三角形インデックス (F, 3)
        boundary_indices: 境界頂点のインデックス
    """
    n = len(boundary)

    # 境界のセグメントを定義
    segments = np.array([[i, (i + 1) % n] for i in range(n)])

    # triangle ライブラリの入力形式
    input_dict = {"vertices": boundary, "segments": segments}

    # 最大面積を自動計算（境界のバウンディングボックスに基づく）
    if max_area is None:
        bbox_size = np.ptp(boundary, axis=0)
        max_area = (bbox_size[0] * bbox_size[1]) / 500

    # 三角形分割（'p': 多角形, 'q': 品質, 'a': 最大面積, 'Y': 境界に頂点を追加しない）
    output = tr.triangulate(input_dict, f"pqYa{max_area}")

    vertices = output["vertices"]
    faces = output["triangles"]

    # 境界頂点のインデックスを特定
    # triangleライブラリは通常、入力された境界点を最初に配置するが、
    # 確実に対応を取るために、各境界点に最も近い頂点を見つける
    boundary_indices = []
    for i in range(n):
        # 各境界点に対して、vertices内で最も近い点を探す
        distances = np.sum((vertices - boundary[i]) ** 2, axis=1)
        closest_idx = np.argmin(distances)
        boundary_indices.append(closest_idx)

    boundary_indices = np.array(boundary_indices)

    # デバッグ情報: 境界点の対応が正しいか確認
    max_dist = np.max(
        [np.linalg.norm(vertices[boundary_indices[i]] - boundary[i]) for i in range(n)]
    )
    if max_dist > 1e-6:
        print(f"  警告: 境界点のマッチング誤差が大きい: {max_dist:.2e}")

    return vertices, faces, boundary_indices


def test_triangulate_boundary(image_path: str, simplify_ratio: float = 0.008, max_area: float = None):
    """
    triangulate_boundary関数のテスト
    
    輪郭を抽出してメッシュを生成し、可視化
    
    Args:
        image_path: 画像ファイルパス
        simplify_ratio: 輪郭簡略化の比率
        max_area: 三角形の最大面積（Noneで自動計算）
    """
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("Testing triangulate_boundary")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Simplify ratio: {simplify_ratio}")
    if max_area is not None:
        print(f"Max triangle area: {max_area}")
    
    # 1. 輪郭抽出
    contour, image = extract_contour_simple(image_path, simplify_ratio)
    print(f"\nContour extraction:")
    print(f"  Contour points: {len(contour)}")
    
    # 輪郭の面積
    contour_area = cv2.contourArea(contour.reshape(-1, 1, 2).astype(np.float32))
    print(f"  Contour area: {contour_area:.2f}")
    
    # 2. メッシュ生成
    print(f"\nGenerating mesh...")
    vertices, faces, boundary_indices = triangulate_boundary(contour, max_area)
    
    print(f"\nMesh generation results:")
    print(f"  Total vertices: {len(vertices)}")
    print(f"  Total faces: {len(faces)}")
    print(f"  Boundary vertices: {len(boundary_indices)}")
    print(f"  Interior vertices: {len(vertices) - len(boundary_indices)}")
    
    # 三角形の面積統計
    triangle_areas = []
    for face in faces:
        v0, v1, v2 = vertices[face]
        area = 0.5 * abs(np.cross(v1 - v0, v2 - v0))
        triangle_areas.append(area)
    
    print(f"\nTriangle area statistics:")
    print(f"  Min area: {np.min(triangle_areas):.2f}")
    print(f"  Max area: {np.max(triangle_areas):.2f}")
    print(f"  Mean area: {np.mean(triangle_areas):.2f}")
    print(f"  Std area: {np.std(triangle_areas):.2f}")
    
    # メッシュ品質の評価
    aspect_ratios = []
    for face in faces:
        v0, v1, v2 = vertices[face]
        # 辺の長さ
        a = np.linalg.norm(v1 - v0)
        b = np.linalg.norm(v2 - v1)
        c = np.linalg.norm(v0 - v2)
        # アスペクト比（最長辺/最短辺）
        aspect_ratio = max(a, b, c) / min(a, b, c)
        aspect_ratios.append(aspect_ratio)
    
    print(f"\nMesh quality:")
    print(f"  Min aspect ratio: {np.min(aspect_ratios):.2f}")
    print(f"  Max aspect ratio: {np.max(aspect_ratios):.2f}")
    print(f"  Mean aspect ratio: {np.mean(aspect_ratios):.2f}")
    
    # Matplotlibで表示
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # 左上: 輪郭のみ
    ax1 = axes[0, 0]
    ax1.set_aspect('equal')
    contour_closed = np.vstack([contour, contour[0]])
    ax1.plot(contour_closed[:, 0], contour_closed[:, 1], 'b-', linewidth=2, label='Boundary')
    ax1.plot(contour[0, 0], contour[0, 1], 'ro', markersize=10, label='Start point')
    ax1.plot(contour[:, 0], contour[:, 1], 'g.', markersize=4, alpha=0.5)
    ax1.invert_yaxis()
    ax1.set_title(f'Original Contour\n{len(contour)} points, Area: {contour_area:.0f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右上: メッシュ（面積で色分け）
    ax2 = axes[0, 1]
    ax2.set_aspect('equal')
    
    # 三角形を面積で色分け
    tri_colors = triangle_areas
    triangulation = plt.matplotlib.tri.Triangulation(vertices[:, 0], vertices[:, 1], faces)
    tcf = ax2.tripcolor(triangulation, tri_colors, cmap='viridis', shading='flat', alpha=0.7)
    ax2.triplot(vertices[:, 0], vertices[:, 1], faces, 'k-', linewidth=0.3, alpha=0.3)
    plt.colorbar(tcf, ax=ax2, label='Triangle Area')
    
    # 境界を強調
    boundary_verts = vertices[boundary_indices]
    boundary_closed = np.vstack([boundary_verts, boundary_verts[0]])
    ax2.plot(boundary_closed[:, 0], boundary_closed[:, 1], 'r-', linewidth=2, label='Boundary')
    
    ax2.invert_yaxis()
    ax2.set_title(f'Mesh colored by Triangle Area\n{len(vertices)} vertices, {len(faces)} triangles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 左下: メッシュ（ワイヤーフレーム + 頂点分類）
    ax3 = axes[1, 0]
    ax3.set_aspect('equal')
    ax3.triplot(vertices[:, 0], vertices[:, 1], faces, 'b-', linewidth=0.5, alpha=0.5)
    
    # 境界頂点を赤で、内部頂点を緑で表示
    interior_mask = np.ones(len(vertices), dtype=bool)
    interior_mask[boundary_indices] = False
    interior_indices = np.where(interior_mask)[0]
    
    ax3.plot(vertices[boundary_indices, 0], vertices[boundary_indices, 1], 
             'ro', markersize=5, label=f'Boundary ({len(boundary_indices)})', alpha=0.8)
    ax3.plot(vertices[interior_indices, 0], vertices[interior_indices, 1], 
             'go', markersize=3, label=f'Interior ({len(interior_indices)})', alpha=0.6)
    
    ax3.invert_yaxis()
    ax3.set_title('Mesh Wireframe with Vertex Classification')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 右下: メッシュ品質（アスペクト比で色分け）
    ax4 = axes[1, 1]
    ax4.set_aspect('equal')
    
    # アスペクト比で色分け
    triangulation = plt.matplotlib.tri.Triangulation(vertices[:, 0], vertices[:, 1], faces)
    tcf = ax4.tripcolor(triangulation, aspect_ratios, cmap='RdYlGn_r', shading='flat', alpha=0.7)
    ax4.triplot(vertices[:, 0], vertices[:, 1], faces, 'k-', linewidth=0.3, alpha=0.3)
    plt.colorbar(tcf, ax=ax4, label='Aspect Ratio (lower is better)')
    
    ax4.invert_yaxis()
    ax4.set_title(f'Mesh Quality (Aspect Ratio)\nMean: {np.mean(aspect_ratios):.2f}')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)


def main():
    """
    メイン関数: コマンドライン引数で画像を指定してテストを実行
    """
    parser = argparse.ArgumentParser(
        description='Test triangulate_boundary: Generate and visualize triangle mesh from image contour',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python triangulate_mesh.py rabbit.png
  
  # With custom simplify ratio (smaller = more detailed contour)
  python triangulate_mesh.py rabbit.png --simplify 0.005
  
  # With custom max triangle area (smaller = finer mesh)
  python triangulate_mesh.py rabbit.png --max-area 50
  
  # Combination
  python triangulate_mesh.py ../yumekawa_animal_usagi.png --simplify 0.008 --max-area 200
        """
    )
    
    parser.add_argument('image', help='Input image file path')
    parser.add_argument('--simplify', type=float, default=0.008,
                       help='Contour simplification ratio (default: 0.008, smaller = more detailed)')
    parser.add_argument('--max-area', type=float, default=None,
                       help='Maximum triangle area for mesh generation (default: auto)')
    
    args = parser.parse_args()
    
    # テスト実行
    test_triangulate_boundary(args.image, args.simplify, args.max_area)


if __name__ == '__main__':
    main()
