"""
PNG画像から輪郭を抽出し、三角形メッシュを生成してテクスチャ付きで表示

使用方法:
    python contour_extract.py image.png
    python contour_extract.py image.png --simplify 0.01
"""

import numpy as np
import cv2
import argparse
import triangle as tr  # pip install triangle
import pyvista as pv  # pip install pyvista


def _load_and_binarize_image(image_path: str) -> tuple:
    """画像を読み込み、二値化する（内部ヘルパー関数）
    
    Args:
        image_path: 画像ファイルパス
    
    Returns:
        image: 元画像
        binary: 二値化画像
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
        # RGBAに変換（extract_contourとの互換性）
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # ノイズ除去
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return image, binary


def ensure_ccw(contour: np.ndarray) -> np.ndarray:
    """反時計回りに統一（符号付き面積で判定）
    
    Args:
        contour: 輪郭点列 (n, 2)
    
    Returns:
        反時計回りに統一された輪郭点列
    """
    # 符号付き面積を計算
    n = len(contour)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += contour[i, 0] * contour[j, 1]
        area -= contour[j, 0] * contour[i, 1]
    area /= 2.0

    # 負なら時計回り → 反転
    if area < 0:
        return contour[::-1].copy()
    return contour


def compute_parallelogram_vectors(
    contour: np.ndarray, theta1: float, theta2: float
) -> tuple:
    """
    閉曲線で囲まれる面積と等しい面積を持つ平行四辺形の2辺のベクトルを計算

    Args:
        contour: 閉曲線の点列 (n, 2)
        theta1: 第1辺の方向（ラジアン、左下から測った角度）
        theta2: 第2辺の方向（ラジアン、左下から測った角度）

    Returns:
        v1: 第1辺のベクトル (2,)
        v2: 第2辺のベクトル (2,)
    """
    # 閉曲線の面積を計算（Shoelace formula）
    n = len(contour)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += contour[i, 0] * contour[j, 1]
        area -= contour[j, 0] * contour[i, 1]
    area = abs(area) / 2.0

    # 2つの角度から単位ベクトルを作成
    u1 = np.array([np.cos(theta1), np.sin(theta1)])
    u2 = np.array([np.cos(theta2), np.sin(theta2)])

    # 2つのベクトル間の角度差
    theta_diff = theta2 - theta1
    sin_theta = abs(np.sin(theta_diff))

    if sin_theta < 1e-10:
        raise ValueError("2つの角度がほぼ平行です。平行四辺形を作成できません。")

    # 平行四辺形の面積 = |v1| * |v2| * sin(theta)
    length = np.sqrt(area / sin_theta)

    v1 = length * u1
    v2 = length * u2

    return v1, v2


def prepare_contour_for_escherization(
    image_path: str,
    simplify_ratio: float = 0.008,
    theta1: float = 0.0,
    theta2: float = np.pi / 2,
    verbose: bool = True
) -> tuple:
    """
    エッシャー化のための輪郭準備（統合関数）
    
    画像から輪郭を抽出し、反時計回りに統一し、平行四辺形ベクトルを計算
    
    Args:
        image_path: 画像ファイルパス
        simplify_ratio: 輪郭簡略化の比率（小さいほど細かい）
        theta1: 第1辺の方向（ラジアン）
        theta2: 第2辺の方向（ラジアン）
        verbose: 詳細情報を出力するか
    
    Returns:
        contour: 反時計回りに統一された輪郭点列 (n, 2)
        image: 元画像
        tran_u: 平行四辺形の第1辺ベクトル
        tran_v: 平行四辺形の第2辺ベクトル
    """
    # 1. 画像読み込みと二値化
    image, binary = _load_and_binarize_image(image_path)
    
    # 2. 輪郭抽出
    contours_raw, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours_raw:
        raise ValueError("輪郭が見つかりません")
    
    # 最大面積の輪郭を選択
    max_contour = max(contours_raw, key=cv2.contourArea)
    
    # 3. Douglas-Peucker簡略化
    perimeter = cv2.arcLength(max_contour, True)
    epsilon = simplify_ratio * perimeter
    simplified = cv2.approxPolyDP(max_contour, epsilon, True)
    contour = simplified.reshape(-1, 2).astype(np.float64)
    
    if verbose:
        print(f"元の輪郭: {len(contour)}点")
    
    # 4. 反時計回りに統一
    contour = ensure_ccw(contour)
    
    # 5. 平行四辺形ベクトルを計算
    tran_u, tran_v = compute_parallelogram_vectors(contour, theta1, theta2)
    
    if verbose:
        print(f"閉曲線の面積: {abs(np.cross(tran_u, tran_v)):.2f}")
        print(f"平行四辺形ベクトル:")
        print(f"  tran_u = ({tran_u[0]:.2f}, {tran_u[1]:.2f})")
        print(f"  tran_v = ({tran_v[0]:.2f}, {tran_v[1]:.2f})")
    
    return contour, image, tran_u, tran_v


def extract_contour(image_path: str, simplify_ratio: float = 0.008) -> tuple:
    """画像から最大の輪郭を抽出

    Args:
        image_path: 画像ファイルパス
        simplify_ratio: 輪郭簡略化の比率（小さいほど細かい）

    Returns:
        contour: 輪郭点列 (n, 2)
        image: 元画像
    """
    # 共通ヘルパーを使用
    image, binary = _load_and_binarize_image(image_path)
    
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


def test_extract_contour(image_path: str, simplify_ratio: float = 0.008):
    """
    extract_contour関数のテスト
    
    輪郭を抽出して、元画像と輪郭の2つのウィンドウで表示
    
    Args:
        image_path: 画像ファイルパス
        simplify_ratio: 輪郭簡略化の比率
    """
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("Testing extract_contour_simple")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Simplify ratio: {simplify_ratio}")
    
    # 輪郭抽出
    contour, image = extract_contour(image_path, simplify_ratio)
    
    print(f"Contour points: {len(contour)}")
    print(f"Image shape: {image.shape}")
    
    # 輪郭面積を計算
    area = cv2.contourArea(contour.reshape(-1, 1, 2).astype(np.float32))
    print(f"Contour area: {area:.2f}")
    
    # 輪郭の周長を計算
    perimeter = cv2.arcLength(contour.reshape(-1, 1, 2).astype(np.float32), True)
    print(f"Contour perimeter: {perimeter:.2f}")
    
    # Matplotlibで表示
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 左: 元画像
    ax1 = axes[0]
    if image.shape[2] == 4:
        # BGRA → RGBA
        image_display = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        image_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(image_display)
    ax1.set_title(f"Original Image\n{image.shape[1]}x{image.shape[0]}")
    ax1.axis('off')
    
    # 右: 輪郭
    ax2 = axes[1]
    ax2.set_aspect('equal')
    
    # 閉曲線として描画
    contour_closed = np.vstack([contour, contour[0]])
    ax2.plot(contour_closed[:, 0], contour_closed[:, 1], 'b-', linewidth=2, label='Contour')
    ax2.plot(contour[0, 0], contour[0, 1], 'ro', markersize=10, label='Start point')
    ax2.plot(contour[:, 0], contour[:, 1], 'g.', markersize=4, alpha=0.5, label=f'Points ({len(contour)})')
    
    # Y軸を反転（画像座標系に合わせる）
    ax2.invert_yaxis()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'Extracted Contour\nArea: {area:.0f}, Points: {len(contour)}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)


def main():
    """
    メイン関数: コマンドライン引数で画像を指定してextract_contour_simpleをテスト
    """
    parser = argparse.ArgumentParser(
        description='Test extract_contour_simple: Extract and visualize image contour',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python contour_extract.py rabbit.png
  python contour_extract.py rabbit.png --simplify 0.01
  python contour_extract.py ../yumekawa_animal_usagi.png --simplify 0.005
        """
    )
    
    parser.add_argument('image', help='Input image file path')
    parser.add_argument('--simplify', type=float, default=0.008,
                       help='Contour simplification ratio (default: 0.008, smaller = more detailed)')
    
    args = parser.parse_args()
    
    # テスト実行
    test_extract_contour(args.image, args.simplify)


if __name__ == '__main__':
    main()


