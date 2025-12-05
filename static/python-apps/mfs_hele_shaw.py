"""
基本解近似解法 (MFS) によるヘレショウ問題（Web版）
基本解近似解法_MFS_ヘレショウ問題_new.py をベースにWeb用に修正
"""

import numpy as np
import math
from moving_curve_new import MovingCurve

# グローバル変数
xlim = 7
ylim = 7
sigma = 1
myu = 1
b_2 = 12 * myu


def G_matrix(x, y, z, N):
    X = np.stack([x] * N, axis=1)
    Y = np.stack([y] * N, axis=0)
    Z = np.stack([z] * N, axis=0)
    G = (
        1.0
        / (2 * math.pi)
        * np.log(np.linalg.norm(X - Y, axis=2) / np.linalg.norm(X - Z, axis=2))
    )
    return G


def H_ij_matrix(x, y, z):
    """
    H_ij = ∇E(X_i - y_j) - ∇E(X_i - z_j)
    E = 1/(2π) * log|X|, ∇E(X) = 1/(2π) * X / |X|^2
    """
    X_i = x[:, np.newaxis, :]
    Y_j = y[np.newaxis, :, :]
    Z_j = z[np.newaxis, :, :]
    
    diff_y = X_i - Y_j
    diff_z = X_i - Z_j
    
    dist_y_sq = np.sum(diff_y ** 2, axis=2, keepdims=True)
    dist_z_sq = np.sum(diff_z ** 2, axis=2, keepdims=True)
    
    H_ij = 1.0 / (2 * math.pi) * (diff_y / dist_y_sq - diff_z / dist_z_sq)
    
    return H_ij


def H_j_vector(H_ij, n, r):
    """H_j = Σ_i (H_ij · n_i) * r_i"""
    H_dot_n = np.sum(H_ij * n[:, np.newaxis, :], axis=2)
    H_j = np.sum(H_dot_n * r[:, np.newaxis], axis=0)
    return H_j


def H_matrix(x, y, z, n, r):
    H_ij = H_ij_matrix(x, y, z)
    H_j = H_j_vector(H_ij, n, r)
    return H_j, H_ij


def P_uv(u, v, y, z, Q_hat):
    """圧力場を計算"""
    u_exp = u[..., np.newaxis]
    v_exp = v[..., np.newaxis]
    
    dist_y_sq = (u_exp - y[:, 0]) ** 2 + (v_exp - y[:, 1]) ** 2
    dist_z_sq = (u_exp - z[:, 0]) ** 2 + (v_exp - z[:, 1]) ** 2
    
    log_terms = np.log(dist_y_sq / dist_z_sq) / 2
    result = Q_hat[0] + 1.0 / (2 * math.pi) * np.sum(Q_hat[1:] * log_terms, axis=-1)
    
    return result


def G_hat_matrix(x, y, z, n, r, N):
    H, H_ij = H_matrix(x, y, z, n, r)
    G = G_matrix(x, y, z, N)
    one = np.ones(N).reshape((N, 1))
    zero = np.zeros(1)
    return np.vstack([np.hstack([zero, H]), np.hstack([one, G])]), H_ij, H


def v_step(curve: MovingCurve, Q_hat, H_ji, n):
    """速度を計算して曲線を更新"""
    Q = Q_hat[1:]
    P_nabra = np.sum(Q[..., np.newaxis, np.newaxis] * H_ji, axis=0)
    v = -1 * b_2 / (12 * myu) * np.sum(P_nabra * n, axis=1)
    curve.vel = np.roll(v, 1)
    curve.uniform_asymptotic()


def MFS(curve: MovingCurve):
    """MFS方程式を解く"""
    N = len(curve._X)
    d = 1 / math.sqrt(N)
    
    phi = sigma * np.roll(curve._k, -1)
    x = np.roll(curve._mid_X, -1, axis=0)
    n = np.roll(curve._norm, -1, axis=0)
    y = x + d * n
    z = y * 1000
    r = np.roll(curve._r, -1)
    
    G_hat, H_ji, H = G_hat_matrix(x, y, z, n, r, N)
    phi_hat = np.concatenate([[0], phi])
    Q_hat = np.linalg.solve(G_hat, phi_hat)
    
    return x, y, z, Q_hat, H_ji, n, phi


def contour_data(y, z, Q_hat, xlim=7, ylim=7, resolution=30):
    """等高線データを取得"""
    x_range = np.linspace(-xlim, xlim, resolution)
    y_range = np.linspace(-ylim, ylim, resolution)
    u, v = np.meshgrid(x_range, y_range)
    P = P_uv(u, v, y, z, Q_hat)
    return u.tolist(), v.tolist(), P.tolist()


def step_simulation(curve: MovingCurve):
    """1ステップ進める"""
    x, y, z, Q_hat, H_ji, n, phi = MFS(curve)
    fn = lambda: v_step(curve, Q_hat, H_ji, n)
    curve.step_RungeKutta4(fn)
    return y, z, Q_hat


def get_curve_data(curve: MovingCurve):
    """曲線データを取得"""
    x, y = curve.xy()
    return x.tolist(), y.tolist()
