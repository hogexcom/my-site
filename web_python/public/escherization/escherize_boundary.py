import numpy as np
import p3
import p1
import p2


def rotate_array(arr: np.ndarray, start_idx: int) -> np.ndarray:
    """配列を回転して start_idx を先頭にする"""
    return np.roll(arr, -start_idx, axis=0)


def subdivide_boundary(contour: np.ndarray, n_target: int) -> tuple:
    """
    線分の長い順から中点を追加して境界点数をNにする

    Args:
        contour: 境界点配列 (n_points, 2)
        N: 目標の境界点数

    Returns:
        adjusted_contour: 調整後の境界点 (N, 2)
    """
    points = contour.copy()
    n_original = len(points)

    print(f"  境界点調整: {n_original}点 → {n_target}点 (N={n_target})")
    # 必要な追加点数
    points_to_add = n_target - len(points)

    for _ in range(points_to_add):
        # 各線分の長さを計算
        n = len(points)
        segment_lengths = []

        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            length = np.linalg.norm(p2 - p1)
            segment_lengths.append((length, i))

        # 最も長い線分を見つける
        segment_lengths.sort(reverse=True)
        longest_length, longest_idx = segment_lengths[0]

        # 最長線分の中点を計算
        p1 = points[longest_idx]
        p2 = points[(longest_idx + 1) % n]
        midpoint = (p1 + p2) / 2.0

        # 中点を挿入
        points = np.insert(points, longest_idx + 1, midpoint, axis=0)

    return points


def resample_contour(contour: np.ndarray, n_target: int) -> np.ndarray:
    """境界を指定頂点数に弧長パラメータでリサンプリング

    Args:
        contour: 境界点配列 (n_points, 2)
        n_target: 目標頂点数

    Returns:
        resampled: リサンプリングされた境界点 (n_target, 2)
    """
    n_boundary = len(contour)

    # 閉曲線の累積弧長
    closed = np.vstack([contour, contour[0:1]])
    diffs = np.diff(closed, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0], np.cumsum(lengths)])
    total_len = cumlen[-1]

    # 等弧長間隔でサンプリング
    target_params = np.linspace(0, total_len, n_target, endpoint=False)

    resampled = np.zeros((n_target, 2))
    for i, t in enumerate(target_params):
        idx = np.searchsorted(cumlen[1:], t)
        if idx >= n_boundary:
            idx = n_boundary - 1

        t0 = cumlen[idx]
        t1 = cumlen[idx + 1]
        if t1 - t0 > 1e-10:
            alpha = (t - t0) / (t1 - t0)
        else:
            alpha = 0.0

        p0 = contour[idx]
        p1 = contour[(idx + 1) % n_boundary]
        resampled[i] = (1 - alpha) * p0 + alpha * p1

    return resampled


def compute_escherization(X: np.ndarray, G: np.ndarray, x_p: np.ndarray):
    """
    エッシャー化の誤差を計算
    誤差 = Σ(エッシャー化後の点 - 元の点)^2
    """
    x = X[:, 0]
    y = X[:, 1]
    w = np.transpose(np.array([x, y])).flatten()

    import warnings

    # print(X.shape, G.shape, x_p.shape, w.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        escherized_boundary: np.ndarray = G @ (np.transpose(G) @ w) + x_p

    escherized_boundary = escherized_boundary.reshape(len(X), 2)

    # 各頂点の移動距離の2乗の合計
    diff = escherized_boundary - X
    error = np.sum(diff**2)
    return escherized_boundary, error


def find_best_escheriztion(
    original: np.ndarray, G: np.ndarray, x_p: np.ndarray
) -> tuple:
    """
    全ての開始点でエッシャー化を試し、最も誤差が小さいものを選択
    """
    n = len(original)
    errors = []
    escherized_boundarys = []

    for start_idx in range(n):
        X_rotated = rotate_array(original, start_idx)
        escherized_boundary, error = compute_escherization(X_rotated, G, x_p)
        escherized_boundarys.append(escherized_boundary)
        errors.append(error)

    best_idx = np.argmin(errors)
    best_error = errors[best_idx]
    best_boundary = escherized_boundarys[best_idx]
    rotated_original = rotate_array(original, best_idx)

    return best_boundary, rotated_original, best_idx, best_error, errors


def adjust_contour(
    contour: np.ndarray, n_target: int, resample: bool = True
) -> np.ndarray:
    if resample == True:
        adjusted = resample_contour(contour, n_target)
    else:
        adjusted = subdivide_boundary(contour, n_target)
    return adjusted


def escherization(
    contour: np.ndarray,
    tiling_pattern: str,
    m: int,
    n: int,
    resample=True,
    trans_u=None,
    trans_v=None,
):
    if tiling_pattern == "P3":
        X = adjust_contour(contour, p3.number_of_vertices(m), resample)
        A, G, b, x_p = p3.kernel(m)
    elif tiling_pattern == "P1":
        X = adjust_contour(contour, p1.number_of_vertices(m, n), resample)
        A, G, b, x_p = p1.kernel(trans_u, trans_v, m, n)
    elif tiling_pattern == "P2":
        X = adjust_contour(contour, p2.number_of_vertices(m, n), resample)
        A, G, b, x_p = p2.kernel(m, n)

    return find_best_escheriztion(X, G, x_p)
