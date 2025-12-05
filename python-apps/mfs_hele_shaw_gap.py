"""
基本解近似解法（MFS）による隙間上昇ヘレショウ問題のWeb版
"""
import numpy as np
import math

pi = math.pi

# パラメータ
xlim = 2
ylim = 2
sigma = 2 * 10**-4
N = 200  # 点の数
d = 1 / math.sqrt(N)


def initial_data(N):
    """初期曲線データ（摂動付き円）"""
    u = np.linspace(0, 1, N+1)
    R = (
        1
        + 0.02
        * (
            np.cos(6 * pi * u)
            + np.sin(14 * pi * u)
            + np.cos(30 * pi * u)
            + np.sin(50 * pi * u)
        )
    )
    X = R[..., np.newaxis] * np.stack([np.cos(2 * pi * u), np.sin(2 * pi * u)], axis=1)
    return X[:-1]


def G_matrix(x, y, z, N):
    """グリーン関数行列"""
    X = np.stack([x] * N, axis=1)
    Y = np.stack([y] * N, axis=0)
    Z = np.stack([z] * N, axis=0)
    G = (
        1.0
        / (2 * math.pi)
        * np.log(np.linalg.norm(X - Y, axis=2) / np.linalg.norm(X - Z, axis=2))
    )
    return G


def H_ji_matrix(x, y, z, N):
    """H行列（勾配の差）"""
    X = np.stack([x] * N, axis=0)
    Y = np.stack([y] * N, axis=1)
    Z = np.stack([z] * N, axis=1)
    return (
        1.0
        / (2 * math.pi)
        * (
            (X - Y) / np.sum((X - Y) ** 2, axis=2)[..., np.newaxis]
            - (X - Z) / np.sum((X - Z) ** 2, axis=2)[..., np.newaxis]
        )
    )


def H_matrix(x, y, z, n, r, N):
    """H行列と法線・弧長の重み付け"""
    H_ji = H_ji_matrix(x, y, z, N)
    nr_i = np.stack([n * r[..., np.newaxis]] * N)
    H_nr_ji = np.sum(H_ji * nr_i, axis=2)
    return np.sum(H_nr_ji, axis=1), H_ji


def P_uv(u, v, y, z, Q_hat):
    """圧力場の計算"""
    sum_val = 0
    for i in range(len(y)):
        sum_val += (
            Q_hat[i + 1]
            * np.log(
                ((u - y[i, 0]) ** 2 + (v - y[i, 1]) ** 2)
                / ((u - z[i, 0]) ** 2 + (v - z[i, 1]) ** 2)
            )
            / 2
        )
    sum_val = Q_hat[0] + 1.0 / (2 * math.pi) * sum_val
    return sum_val


def G_hat_matrix(x, y, z, n, r, N):
    """拡張グリーン関数行列"""
    H, H_ji = H_matrix(x, y, z, n, r, N)
    G = G_matrix(x, y, z, N)
    one = np.ones(N).reshape((N, 1))
    zero = np.zeros(1)
    return np.vstack([np.hstack([zero, H]), np.hstack([one, G])]), H_ji, H


def b_dot(t):
    """b(t)の時間微分"""
    return math.exp(t)


def b(t):
    """隙間幅関数 b(t)"""
    return math.exp(t)


def phi_hat(X_, t, curve):
    """境界条件"""
    k = np.roll(curve._k, -1)
    phi = sigma * k - b_dot(t) / (4 * b(t) ** 3) * np.sum(X_ * X_, axis=1)
    _phi_hat = np.concatenate([[0], phi])
    return _phi_hat, phi


def MFS(curve):
    """MFSの主計算"""
    N = len(curve._X)
    n = np.roll(curve._norm, -1, axis=0)
    x = np.roll(curve._mid_X, -1, axis=0)
    y = x + d * n
    z = (
        np.stack(
            [
                np.cos(2 * pi * np.arange(1, N + 1) / N),
                np.sin(2 * pi * np.arange(1, N + 1) / N),
            ],
            axis=1,
        )
        * 1000
    )
    r = np.roll(curve._r, -1, axis=0)
    G_hat, H_ji, H = G_hat_matrix(x, y, z, n, r, N)
    _phi_hat, phi = phi_hat(x, curve.elapsed_time, curve)
    Q_hat = np.linalg.solve(G_hat, _phi_hat)
    return x, y, z, Q_hat, H_ji, n, phi


def v_step(curve, Q_hat, H_ji, X_, n):
    """速度場の計算とcurveの更新"""
    Q = Q_hat[1:]
    P_nabra = np.sum(Q[..., np.newaxis, np.newaxis] * H_ji, axis=0)
    t = curve.elapsed_time
    v = -1 * b(t) ** 2 * np.sum(P_nabra * n, axis=1) - b_dot(t) / (2 * b(t)) * np.sum(
        X_ * n, axis=1
    )
    curve.vel = np.roll(v, 1)
    curve.uniform_asymptotic()


def step_simulation(curve):
    """1ステップのシミュレーション"""
    x, y, z, Q_hat, H_ji, n, phi = MFS(curve)
    fn = lambda: v_step(curve, Q_hat, H_ji, x, n)
    curve.step_RungeKutta4(fn)
    return y, z, Q_hat


def contour_data(y, z, Q_hat, xlim=2, ylim=2, resolution=30):
    """等高線データを計算"""
    x_range = np.linspace(-xlim, xlim, resolution)
    y_range = np.linspace(-ylim, ylim, resolution)
    u, v = np.meshgrid(x_range, y_range)
    P = P_uv(u, v, y, z, Q_hat)
    return u.tolist(), v.tolist(), P.tolist()


def get_curve_data(curve):
    """曲線の座標を取得"""
    x, y = curve.xy()
    return x.tolist(), y.tolist()
