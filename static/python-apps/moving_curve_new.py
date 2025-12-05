import numpy as np
import math
from functools import cache
import random
from typing import Callable, Optional
from numpy.typing import NDArray


"""
点のインデックスは教科書通り
点は0から始まり、Nで終わる。
点0は点Nと同じ、点1は点N+1と同じであるので、
X[N] = X[0], X[N+1] = X[1] となる。
ただし、循環しない配列、nyuは前後にバッファを設ける。
教科書通りのインデックスを配列に用いるため、
辺[0]は辺Nと同じで、辺[1]は辺1と同じである。
Xの配列サイズはNである。
T[1]はX[1:0]の辺、T[0]とT[N]はX[0:N-1]の辺
"""


class MovingCurve:
    def __init__(self, X: NDArray[np.float64], epsilon: float = 0.7) -> None:
        N: int = len(X)
        self._N: int = -1
        dt: float = 0.1 / N**2
        ω: float = 10 * N
        self.dt: float = dt
        self.ω: float = ω
        self.epsilon: float = epsilon
        self.elapsed_time: float = 0
        self.set_X(X)
        self.update_point_insertion()

    @classmethod
    def initialize_ellipse(
        cls, N: int, r: float = 1, a: float = 1, b: float = 1, epsilon: float = 0.7
    ) -> "MovingCurve":
        theta = np.linspace(0, math.pi * 2, N + 1)
        theta = theta[:-1].copy()
        x = np.cos(theta) * r * a
        y = np.sin(theta) * r * b
        X = np.transpose(np.stack([x, y]))
        instance = cls(X, epsilon=epsilon)
        return instance

    @classmethod
    def initializeCurvatureTest(cls) -> "MovingCurve":
        count = 10
        y = np.linspace(-4, 5, count)
        x = np.ones(count) * -5
        l1 = np.transpose(np.stack([x, y]))
        theta = np.linspace(math.pi, math.pi / 2, count)
        x = np.cos(theta) - 4
        y = np.sin(theta) + 5
        l2 = np.transpose(np.stack([x, y]))[1:]
        x = np.linspace(-4, 4, count)
        y = np.ones(count) * 6
        l3 = np.transpose(np.stack([x, y]))[1:]
        theta = np.linspace(math.pi / 2, 0, count)
        x = np.cos(theta) + 4
        y = np.sin(theta) + 5
        l4 = np.transpose(np.stack([x, y]))[1:]
        y = np.linspace(5, -4, count)
        x = np.ones(count) * 5
        l5 = np.transpose(np.stack([x, y]))[1:]
        theta = np.linspace(0, -math.pi / 2, count)
        x = np.cos(theta) + 4
        y = np.sin(theta) - 4
        l6 = np.transpose(np.stack([x, y]))[1:]
        x = np.linspace(4, -4, count)
        y = np.ones(count) * -5
        l7 = np.transpose(np.stack([x, y]))[1:]
        theta = np.linspace(-math.pi / 2, -math.pi, count)
        x = np.cos(theta) - 4
        y = np.sin(theta) - 4
        l8 = np.transpose(np.stack([x, y]))[1:-1]
        xy = np.append(l1, l2, axis=0)
        xy = np.append(xy, l3, axis=0)
        xy = np.append(xy, l4, axis=0)
        xy = np.append(xy, l5, axis=0)
        xy = np.append(xy, l6, axis=0)
        xy = np.append(xy, l7, axis=0)
        xy = np.append(xy, l8, axis=0)
        # self._N = len(xy)
        instance = cls(xy)
        return instance

    @classmethod
    def initialize_uniform_XY(
        cls, N: int, r: float = 0.5, a: float = 1, b: float = 1
    ) -> "MovingCurve":
        """等間隔な弧長で配置された楕円の頂点でMovingCurveを初期化

        Args:
            N: 頂点数
            r: 楕円のスケール
            a: x方向のアスペクト比
            b: y方向のアスペクト比

        Returns:
            初期化されたMovingCurveインスタンス
        """
        u = []
        u_prev = 0
        two_pi = math.pi * 2
        dw = 1.0 / N

        # 楕円の周長を数値積分で計算
        u_samples = np.linspace(0, 1, N * 2)
        g = (
            r
            * two_pi
            * np.sqrt(
                a**2 * np.sin(two_pi * u_samples) ** 2
                + b**2 * np.cos(two_pi * u_samples) ** 2
            )
        )
        L = 0
        for i in range(1, N * 2):
            du = u_samples[i] - u_samples[i - 1]
            L += (g[i] + g[i - 1]) / 2 * du

        # 等弧長で頂点を配置
        for i in range(0, N + 1):
            g = (
                r
                * two_pi
                * math.sqrt(
                    a**2 * math.sin(two_pi * u_prev) ** 2
                    + b**2 * math.cos(two_pi * u_prev) ** 2
                )
            )
            u_next = u_prev + dw * (L / g)
            u.append(u_next)
            u_prev = u_next
        u = np.array(u)
        x = r * a * np.cos(two_pi * u)
        y = r * b * np.sin(two_pi * u)
        X = np.transpose(np.stack([x, y]))[:-1]  # 最後の点を除く（周期境界のため）

        return cls(X)

    def initialize_X(self, X: NDArray[np.float64]) -> None:
        self.set_X(X)
        self.update_point_insertion()

    def update_point_insertion(self) -> None:
        self.point_insertion = {
            "L": self._L,
            "c0": self.dt / (self._L / self._N) ** 2,
        }

    def add_noise(self) -> None:
        # 各頂点での平均辺長: (r[i] + r[i+1]) / 2
        r_plus = np.roll(self._r, -1)
        avg_r = (self._r + r_plus) / 2
        # ランダムな極座標
        rho = avg_r * 0.05 * np.random.random(self._N)
        theta = np.pi * 2 * np.random.random(self._N)
        # ノイズベクトル
        noise = np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])
        XN = self._X + noise
        self.set_X(XN)

    def set_X(self, X: NDArray[np.float64]) -> None:
        N = len(X)
        if N != self._N:
            print("頂点数が変化しました。配列を再初期化します。", N)
            self._N: int = N
            # 辺上の法線速度
            self._vel: NDArray[np.float64] = np.zeros(N)
            # 頂点上の法線速度
            # self._Vel: NDArray[np.float64] = np.zeros(N)
            # 頂点上の接線速度
            self._W: NDArray[np.float64] = np.zeros(N)
            # self._dX_dt: NDArray[np.float64] = np.zeros(N * 2).reshape(N, 2)  # Xの時間微分
            # self._rho_phi: NDArray[np.float64] = np.zeros(N)
        self._X = X
        # 遅延初期化される配列のキャッシュを無効化
        self._cached_kF: Optional[NDArray[np.float64]] = None
        self._cached_K: Optional[NDArray[np.float64]] = None
        self._cached_k_hat: Optional[NDArray[np.float64]] = None
        self._cached_kF_hat: Optional[NDArray[np.float64]] = None
        self._cached_k_mean: Optional[NDArray[np.float64]] = None
        self._cached_D_s_k: Optional[NDArray[np.float64]] = None
        self._cached_D_ss_d_v: Optional[NDArray[np.float64]] = None
        self._cached_D_s_c_v: Optional[NDArray[np.float64]] = None
        self._cached_D_ss_d_v: Optional[NDArray[np.float64]] = None
        self._cached_D_s_v_hat_v: Optional[NDArray[np.float64]] = None
        self._cached_kv: Optional[NDArray[np.float64]] = None
        self._compute_geometry_arrays()

    def _compute_geometry_arrays(self) -> None:
        """全頂点の幾何量を一括計算して拡張配列に格納"""
        N = self._N

        # 2. 辺ベクトル Γ[i] = X[i] - X[i-1] をスライスで計算
        self._Γ = self._X - np.roll(self._X, 1, axis=0)  # 一気に差分

        # 3. 辺の長さ r[i] = |Γ[i]|
        self._r = np.linalg.norm(self._Γ, axis=1)
        # ゼロ除算回避
        self._r = np.where(self._r < 1e-12, 1e-12, self._r)

        # 4. 接線ベクトル t[i] = Γ[i] / r[i]
        self._t = self._Γ / self._r[:, np.newaxis]

        # 5. 累積回転角 nyu の計算（ベクトル化）
        # 0 ... N - 1, N, N + 1, N + 2, -1 の辺の値を保持する拡張配列
        self._nyu_ext: NDArray[np.float64] = np.zeros(N + 4)
        # nyu[1] の初期値
        nyu_1 = self.acos(self._t[1, 0]) * (1 if self._t[1, 1] >= 0 else -1)
        t_i_indices = np.arange(1, N + 2)  # [1, 2, ..., N+1]
        t_i_next_indices = np.arange(2, N + 3)  # [2, 3, ..., N+2]
        t_i = np.take(self._t, t_i_indices, axis=0, mode="wrap")
        t_i_next = np.take(self._t, t_i_next_indices, axis=0, mode="wrap")
        # 各要素ごとの内積: t[i] · t[i+1]
        I = np.sum(t_i * t_i_next, axis=1)
        # 各要素ごとの行列式: det([t[i], t[i+1]])
        D = t_i[:, 0] * t_i_next[:, 1] - t_i[:, 1] * t_i_next[:, 0]
        # 累積和
        angle_increments = np.sign(D) * np.arccos(np.clip(I, -1.0, 1.0))
        self._nyu_ext[1 : N + 3] = np.concatenate(
            [[nyu_1], nyu_1 + np.cumsum(angle_increments)]
        )
        self._nyu_ext[0] = nyu_1 - (self._nyu_ext[N + 1] - self._nyu_ext[N])
        self._nyu_ext[N + 3] = self._nyu_ext[0] - (
            self._nyu_ext[N] - self._nyu_ext[N - 1]
        )

        self._phi = self._nyu_ext[1 : N + 1] - self._nyu_ext[0:N]
        self._phi_hat = (self._phi + np.roll(self._phi, 1)) / 2
        # # 総回転角
        # turn = self._nyu_ext[N] - self._nyu_ext[0]

        # 6. cosi, sini, tani （phi/2 の三角関数）
        self._cosi = np.cos(self._phi / 2)
        self._sini = np.sin(self._phi / 2)
        self._tani = np.tan(self._phi / 2)

        # ゼロ除算回避
        self._cosi = np.where(np.abs(self._cosi) < 1e-12, 1e-12, self._cosi)

        # 7. 平均接線 T[i] = (cos(nyu_hat), sin(nyu_hat))
        self._nyu_hat = self._nyu_ext[0:N] + 0.5 * (self._phi)
        self._T = np.column_stack([np.cos(self._nyu_hat), np.sin(self._nyu_hat)])

        # 8. 法線 N[i] = (sin(nyu_hat), -cos(nyu_hat))
        self._Norm = np.column_stack([np.sin(self._nyu_hat), -np.cos(self._nyu_hat)])

        # 9. 曲率 k[i] = (tan(phi[i]/2) + tan(phi[i-1]/2)) / r[i]
        self._k = (self._tani + np.roll(self._tani, 1)) / self._r
        self._mid_X = self._X - self._Γ / 2
        self._r_hat = (self._r + np.roll(self._r, -1)) / 2
        # 法線ベクトル n[i] = (-t[i].y, t[i].x) = 接線を左に90度回転
        self._norm = -1 * np.column_stack([-self._t[:, 1], self._t[:, 0]])
        self._L = np.sum(self._r)

    def acos(self, v):
        _v = v
        if _v < -1:
            _v = -1
        elif _v > 1:
            _v = 1
        return math.acos(_v)

    def L(self) -> float:
        """曲線の総弧長を返す"""
        return self._L

    @property
    def kF(self) -> NDArray[np.float64]:
        """遅延初期化: kF[i] = phi_hat[i] / r[i]"""
        if self._cached_kF is None:
            self._cached_kF = self._phi_hat / self._r
        return self._cached_kF

    @property
    def K(self) -> NDArray[np.float64]:
        """遅延初期化: K[i] = (k[i] + k[i+1]) / (2 * cosi[i])"""
        if self._cached_K is None:
            self._cached_K = (self._k + np.roll(self._k, -1)) / (2 * self._cosi)
        return self._cached_K

    @property
    def k_hat(self) -> NDArray[np.float64]:
        """遅延初期化: k_hat[i] = (2 * sini[i]) / r_hat[i]"""
        if self._cached_k_hat is None:
            self._cached_k_hat = (2 * self._sini) / self._r_hat
        return self._cached_k_hat

    @property
    def kF_hat(self) -> NDArray[np.float64]:
        """遅延初期化: kF_hat[i] = phi[i] / r_hat[i]"""
        if self._cached_kF_hat is None:
            self._cached_kF_hat = self._phi / self._r_hat
        return self._cached_kF_hat

    @property
    def k_mean(self) -> NDArray[np.float64]:
        """遅延初期化: k_mean[i] = (k[i] + k[i+1]) / 2"""
        if self._cached_k_mean is None:
            self._cached_k_mean = (self._k + np.roll(self._k, -1)) / 2
        return self._cached_k_mean

    @property
    def vel(self) -> NDArray[np.float64]:
        """辺上の法線速度"""
        return self._vel

    @vel.setter
    def vel(self, vel: NDArray[np.float64]) -> None:
        """辺上の法線速度を設定し、頂点上の法線速度を自動計算"""
        self._vel = vel
        self._Vel = (np.roll(vel, -1) + vel) / (2 * self._cosi)

    # D_s_e, D_s_v_hat は汎用関数なので残す（使われていない場合はコメントアウト）
    # def D_s_e(self, fn, i):
    #     return (fn(i) - fn(i-1)) / self.r(i)

    # def D_s_v_hat(self, fn, i):
    #     return (fn(i+1) - fn(i)) / self.r_hat(i)

    @property
    def D_s_k(self) -> NDArray[np.float64]:
        """遅延初期化: D_s_k[i] = (a - b) / r[i]
        a = (k[i] + k[i+1]) / (2 * cosi[i]^2)
        b = (k[i-1] + k[i]) / (2 * cosi[i-1]^2)
        """
        if self._cached_D_s_k is None:
            k_plus = np.roll(self._k, -1)
            k_minus = np.roll(self._k, 1)
            cosi_minus = np.roll(self._cosi, 1)
            a = (self._k + k_plus) / (2 * self._cosi**2)
            b = (k_minus + self._k) / (2 * cosi_minus**2)
            self._cached_D_s_k = (a - b) / self._r
        return self._cached_D_s_k

    @property
    def D_ss_k(self) -> NDArray[np.float64]:
        """遅延初期化: D_ss_k[i] = (D_s_k[i+1] - D_s_k[i-1]) / (2 * r[i])"""
        if self._cached_D_ss_d_v is None:
            D_s_k_arr = self.D_s_k
            D_s_k_plus = np.roll(D_s_k_arr, -1)
            D_s_k_minus = np.roll(D_s_k_arr, 1)
            self._cached_D_ss_d_v = (D_s_k_plus - D_s_k_minus) / (2 * self._r)
        return self._cached_D_ss_d_v

    @property
    def D_s_c_v(self) -> NDArray[np.float64]:
        """遅延初期化: D_s_c_v[i] = (v[i+1] - v[i-1]) / (2 * r[i])"""
        if self._cached_D_s_c_v is None:
            v_plus = np.roll(self._vel, -1)
            v_minus = np.roll(self._vel, 1)
            self._cached_D_s_c_v = (v_plus - v_minus) / (2 * self._r)
        return self._cached_D_s_c_v

    # D_s_d は汎用関数なので個別実装は難しい（必要に応じて後で実装）
    # def D_s_d(self, fn, i):
    #     return (
    #         (fn(i + 1) - fn(i)) / (2 * self.cosi(i) ** 2)
    #         + (fn(i) - fn(i - 1)) / (2 * self.cosi(i - 1) ** 2)
    #     ) / self.r(i)

    # D_ss_d_v は D_s_d(D_s_c_v) の合成なので、D_s_c_v を使って直接計算
    @property
    def D_ss_d_v(self) -> NDArray[np.float64]:
        if self._cached_D_ss_d_v is None:
            D_s_c_v_arr = self.D_s_c_v
            D_s_c_v_plus = np.roll(D_s_c_v_arr, -1)
            D_s_c_v_minus = np.roll(D_s_c_v_arr, 1)
            self._cached_D_ss_d_v = (
                ((D_s_c_v_plus - D_s_c_v_arr) / (2 * self._cosi**2))
                + ((D_s_c_v_arr - D_s_c_v_minus) / (2 * np.roll(self._cosi, 1) ** 2))
            ) / self._r
        return self._cached_D_ss_d_v

    @property
    def D_s_v_hat_v(self) -> NDArray[np.float64]:
        """遅延初期化: D_s_v_hat_v[i] = (v[i+1] - v[i]) / r_hat[i]"""
        if self._cached_D_s_v_hat_v is None:
            v_plus = np.roll(self._vel, -1)
            self._cached_D_s_v_hat_v = (v_plus - self._vel) / self._r_hat
        return self._cached_D_s_v_hat_v

    # D_s_e は汎用関数として残す（配列化は困難）
    # def D_s_e(self, fn, i):  # p186
    #     return (fn(i) - fn(i - 1)) / self.r(i)

    @property
    def kv(self) -> NDArray[np.float64]:
        """遅延初期化: 曲率重み付き平均
        kv[i] = (tani[i]*v[i+1] + (tani[i]+tani[i-1])*v[i] + tani[i-1]*v[i-1]) / (2*r[i])
        """
        if self._cached_kv is None:
            v_plus = np.roll(self._vel, -1)
            v_minus = np.roll(self._vel, 1)
            tani_minus = np.roll(self._tani, 1)
            self._cached_kv = (
                self._tani * v_plus
                + (self._tani + tani_minus) * self._vel
                + tani_minus * v_minus
            ) / (2 * self._r)
        return self._cached_kv

    def step_W_uniform(self):
        self.uniform_asymptotic()
        W = np.column_stack([self._W, self._W])
        XN = self._X + self.dt * (W * self._T)
        self.set_X(XN)

    def step_W_curvature(self):
        self.curvature_asymptotic()
        W = np.column_stack([self._W, self._W])
        XN = self._X + self.dt * (W * self._T)
        self.set_X(XN)

    def step(self, fn):
        fn()
        V = np.column_stack([self._Vel, self._Vel])
        W = np.column_stack([self._W, self._W])
        return V * self._Norm + W * self._T

    def step_Eular(self, fn):
        NT = self.step(fn)
        self._dX_dt = NT
        XN = self._X + self.dt * NT
        self.set_X(XN)
        self.after_step()

    def step_RungeKutta2(self, fn):
        X = self._X
        s1 = self.step(fn)
        XN = self._X + s1 * self.dt
        self.set_X(XN)
        s2 = self.step(fn)
        dt_X = 0.5 * (s1 + s2)
        XN = X + dt_X * self.dt
        self._dX_dt = dt_X
        self.set_X(XN)
        self.after_step()

    def step_RungeKutta4(self, fn):
        X = self._X
        k1 = self.step(fn)
        XN = X + k1 * self.dt / 2
        self.set_X(XN)
        k2 = self.step(fn)
        XN = X + k2 * self.dt / 2
        self.set_X(XN)
        k3 = self.step(fn)
        XN = X + k3 * self.dt
        self.set_X(XN)
        k4 = self.step(fn)
        dt_X = 1.0 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        XN = X + dt_X * self.dt
        self._dX_dt = dt_X
        self.set_X(XN)
        self.after_step()

    def after_step(self):
        # self.point_insert()
        self.elapsed_time += self.dt

    def point_insert(self):
        if not self._L > self.point_insertion["L"] * 2:
            return
        #     self.dt = self.point_insertion['c0'] * (self.L() / (2*self._N))
        
        print("点の挿入を行います。頂点数:", self._N, "弧長:", self._L)

        # XN = [self.X(0)]
        # for i in range(1, self._N + 1):
        #     newX = (
        #         (self.X(i - 1) + self.X(i)) / 2
        #         + 5 * (self.T(i - 1) - self.T(i)) / 32
        #         - (self.k_hat(i - 1) * self.N(i - 1) + self.k_hat(i) * self.N(i)) / 64
        #     )
        #     XN.append(newX)
        #     XN.append(self.X(i))
        # XN = XN[:-1]
        # XN = np.stack(XN)

        # 配列演算で中点を計算
        X_current = self._X  # X[i]
        X_prev = np.roll(self._X, 1, axis=0)  # X[i-1]
        T_current = self._T  # T[i]
        T_prev = np.roll(self._T, 1, axis=0)  # T[i-1]
        N_current = self._Norm  # N[i]
        N_prev = np.roll(self._Norm, 1, axis=0)  # N[i-1]
        k_hat_current = self.k_hat  # k_hat[i]
        k_hat_prev = np.roll(self.k_hat, 1)  # k_hat[i-1]

        # 各辺の中点を計算 (N個の点)
        midpoints = (
            (X_prev + X_current) / 2
            + 5 * (T_prev - T_current) / 32
            - (
                k_hat_prev[:, np.newaxis] * N_prev
                + k_hat_current[:, np.newaxis] * N_current
            )
            / 64
        )
        midpoints = np.roll(midpoints, -1, axis=0)  # 中点を辺の始点側にシフト

        # 元の点と中点を交互に配置: [X[0], mid[0], X[1], mid[1], ..., X[N-1], mid[N-1]]
        # reshapeとravel()で交互配置
        XN = np.empty((self._N * 2, 2), dtype=np.float64)
        XN[0::2] = self._X  # 偶数インデックスに元の点
        XN[1::2] = midpoints  # 奇数インデックスに中点

        self.set_X(XN)
        self.update_point_insertion()
        # self._rho_phi = np.zeros(self._N + 1)

    def uniform_asymptotic(self) -> None:
        # psy = [0, 0]
        L = self._L
        N = self._N
        # for i in range(1, N + 1):
        #     L_dot += self.k(i) * self.v(i) * self.r(i)
        L_dot = np.sum(
            np.roll(self._k, -1) * np.roll(self._vel, -1) * np.roll(self._r, -1)
        )
        # for i in range(2, N + 1):
        #     psy_ = (
        #         L_dot / N
        #         - self.V(i) * self.sini(i)
        #         - self.V(i - 1) * self.sini(i - 1)
        #         + (L / N - self.r(i)) * self.ω
        #     )
        #     psy.append(psy_)
        psy = (
            L_dot / N
            - np.roll(self._Vel, -2) * np.roll(self._sini, -2)
            - np.roll(self._Vel, -1) * np.roll(self._sini, -1)
            + (L / N - np.roll(self._r, -2)) * self.ω
        )[:-1]
        psy = np.concatenate(([0, 0], psy))
        # PSY = [0]
        # for i in range(1, N + 1):
        #     psy_sum = 0
        #     for j in range(1, i + 1):
        #         psy_sum += psy[j]
        #     PSY.append(psy_sum)
        PSY = np.concatenate(([0], np.cumsum(psy[1:])))
        # c1 = 0
        # c2 = 0
        # for i in range(1, N + 1):
        #     c1 += PSY[i] / self.cosi(i)
        #     c2 += 1 / self.cosi(i)
        # c = -1 * c1 / c2
        c = -1 * np.sum(PSY[1:] / np.roll(self._cosi, -1)) / np.sum(1 / np.roll(self._cosi, -1))
        # W = [0.0]
        # for i in range(1, N + 1):
        #     w = (PSY[i] + c) / self.cosi(i)
        #     if i == N:
        #         W[0] = w
        #     else:
        #         W.append(w)
        W = (PSY[1:] + c) / np.roll(self._cosi, -1)
        W = np.concatenate(([W[-1]], W[:-1]))

        self._W = np.array(W)

    def curvature_asymptotic(self) -> None:
        N = self._N
        epsilon = self.epsilon
        # k_fn = self.k
        # k_hat_fn = self.k_hat
        # vphi = lambda k: 1 - epsilon + epsilon * math.sqrt(1 - epsilon + epsilon * k**2)
        # vphi_dash = (
        #     lambda k: (0.5 * epsilon / math.sqrt(1 - epsilon + epsilon * k**2))
        #     * 2
        #     * epsilon
        #     * k
        # )
        # vphi = lambda k: 1 - epsilon + epsilon * math.sqrt(1 - epsilon + epsilon * math.fabs(k))
        # vphi_dash = lambda k: (0.5*epsilon / math.sqrt(1-epsilon+epsilon*k**2)) * 2*epsilon if k >= 0 else -1 * (0.5*epsilon / math.sqrt(1-epsilon+epsilon*k**2)) * 2*epsilon
        # vphi = lambda k: 1 - epsilon + epsilon * math.fabs(k)
        # vphi_dash = lambda k: 1 if k >= 0 else -1
        # vphi = lambda k: 1 - epsilon + epsilon * (math.fabs(k) ** 2)
        # vphi_dash = lambda k: 2 * k
        # scaler = 0.5
        # vphi = lambda k: -epsilon + epsilon * math.exp(scaler * math.fabs(k)) + 0.1
        # vphi_dash = lambda k: epsilon * scaler * math.fabs(k) * math.exp(math.fabs(k)) if k >= 0 else -1 * epsilon * scaler * math.fabs(k) * math.exp(math.fabs(k))
        # vphi = lambda k: 1 - epsilon + epsilon * math.log(1 + math.fabs(k) ** 4)
        # vphi_dash = lambda k: epsilon / (1 + math.fabs(k) ** 4) * 4 * math.fabs(k) ** 3
        # シグモイド関数
        b = 10
        # vphi = (
        #     lambda k: 1 - epsilon + epsilon / (1 + math.exp(-b * (math.fabs(k) - 0.5)))
        # )
        # vphi_dash = (
        #     lambda k: -epsilon
        #     / (1 + math.exp(-b * (math.fabs(k) - 0.5))) ** 2
        #     * b**2
        #     * (math.fabs(k) - 0.5)
        #     * math.exp(-b * (math.fabs(k) - 0.5))
        #     * (1 if k >= 0 else -1)
        # )
        vphi_dash = (
            -epsilon
            / (1 + np.exp(-b * (np.fabs(self._k) - 0.5))) ** 2
            * b**2
            * (np.fabs(self._k) - 0.5)
            * np.exp(-b * (np.fabs(self._k) - 0.5))
            * np.where(self._k >= 0, 1, -1)
        )
        # vphi_k = [0]
        # vphi_k_hat = [0]
        # for i in range(1, N + 1):
        #     k_value = vphi(k_fn(i))
        #     k_hat_value = vphi(k_hat_fn(i))
        #     vphi_k.append(k_value)
        #     vphi_k_hat.append(k_hat_value)
        #     if i == N:
        #         vphi_k[0] = k_value
        #         vphi_k_hat[0] = k_hat_value
        vphi_k_plus = 1 - epsilon + epsilon / (1 + np.exp(-b * (np.fabs(self._k) - 0.5)))
        vphi_k = np.roll(vphi_k_plus, 1)
        vphi_k_hat_plus = 1 - epsilon + epsilon / (1 + np.exp(-b * (np.fabs(self.k_hat) - 0.5)))
        vphi_k_hat = np.roll(vphi_k_hat_plus, 1)
        # f = [0]
        # for i in range(1, N + 1):
        #     if not k_fn == self.k:
        #         # Vss = (
        #         #     ((self.v(i + 1) - self.v(i)) / self.r_hat(i))
        #         #     - ((self.v(i) - self.v(i - 1)) / self.r_hat(i - 1))
        #         # ) / self.r(i)
        #         Vss = self.D_s_e(self.D_s_v_hat_v, i)
        #         _f = vphi_k[i] * k_fn(i) * self.v(i) - vphi_dash(k_fn(i)) * (
        #             Vss + k_fn(i) ** 2 * self.v(i)
        #         )
        #     else:
        #         Vss = self.D_ss_d_v(i)
        #         _f = vphi_k[i] * k_fn(i) * self.v(i) - vphi_dash(k_fn(i)) * (
        #             Vss + k_fn(i) * self._kv_(i)
        #         )
        #     f.append(_f)
        f = np.zeros(N + 1)
        Vss = self.D_ss_d_v
        f[1:] = np.roll(
            vphi_k * self._k * self._vel
            - vphi_dash * (Vss + self._k * self.kv)
        , -1)
        f[0] = f[N]
        # 平均値計算
        L = self._L
        # f_mean = 0
        # vphi_mean = 0
        # for i in range(1, N + 1):
        #     f_mean += f[i] * self.r(i)
        #     vphi_mean += vphi_k[i] * self.r(i)
        # f_mean /= L
        # vphi_mean /= L
        r_plus = np.roll(self._r, -1)
        f_mean = np.sum(f[1:] * r_plus) / L
        vphi_mean = np.sum(vphi_k_plus * r_plus) / L
        # b = [0]
        # for i in range(1, N + 1):
        #     _b = (
        #         vphi_k[i]
        #         * self.r(i)
        #         * (
        #             f_mean / vphi_mean
        #             - f[i] / vphi_k[i]
        #             + ((L / (N * self.r(i))) * (vphi_mean / vphi_k[i]) - 1) * self.ω
        #         )
        #     )
        #     b.append(_b)
        b = np.zeros(N + 1)
        b[1:] = (
            vphi_k_plus
            * np.roll(self._r, -1)
            * (
                f_mean / vphi_mean
                - f[1:] / vphi_k_plus
                + ((L / (N * r_plus)) * (vphi_mean / vphi_k_plus) - 1) * self.ω
            )
        )
        # sum2 = 0
        # accum_b = 0
        # for i in range(2, N):
        #     accum_b += b[i]
        #     sum2 += accum_b * self.r_hat(i)
        # sum2 += 0.5 * (-b[1] + accum_b + b[N]) * self.r_hat(N)
        accum_b = np.cumsum(b[2:N])
        sum2 = np.sum(accum_b * self._r_hat[2:N])
        sum2 += 0.5 * (-b[1] + accum_b[-1] + b[N]) * self._r_hat[0]
        W1 = -1 * sum2 / (vphi_k_hat[1] * L)
        # W = [0, W1]
        # for i in range(2, N + 1):
        #     _W = (vphi_k_hat[i - 1] * W[i - 1] + b[i]) / vphi_k_hat[i]
        #     if i == N:
        #         W[0] = _W
        #     else:
        #         W.append(_W)
        W = np.zeros(N)
        W[1] = W1
        for i in range(2, N):
            W[i] = (vphi_k_hat[i - 1] * W[i - 1] + b[i]) / vphi_k_hat[i]
        W[0] = (vphi_k_hat[- 1] * W[- 1] + b[0]) / vphi_k_hat[0]  # 周期境界条件
        self._W = W

    def xy(self):
        """閉曲線として描画するためのxy座標を取得（最後に最初の点を追加）"""
        N = self._N
        # 事前割り当てでconcatenateを回避
        x = np.empty(N + 1, dtype=np.float64)
        y = np.empty(N + 1, dtype=np.float64)
        x[:N] = self._X[:, 0]
        y[:N] = self._X[:, 1]
        x[N] = self._X[0, 0]
        y[N] = self._X[0, 1]
        return (x, y)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    # 初期化
    N = 100
    # curve = MovingCurve.initialize_uniform_XY(N, r=0.5, a=3, b=1)
    curve = MovingCurve.initializeCurvatureTest()
    
    # 初期速度を設定（曲率に比例した法線速度）
    curve.vel = -curve._k * 0.1
    
    # アニメーション設定
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.grid(True, alpha=0.3)
    
    # プロット要素の初期化
    xy = curve.xy()
    line, = ax.plot(xy[0], xy[1], 'b-o', markersize=4, linewidth=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    points_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=12)
    
    def update(frame):
        """各フレームでの更新処理"""
        # step_W_uniformを実行して頂点を移動
        # curve.step_W_uniform()
        curve.step_W_curvature()
        
        # 更新された座標を取得
        xy = curve.xy()
        line.set_data(xy[0], xy[1])
        
        # 情報表示を更新
        time_text.set_text(f'Frame: {frame}')
        points_text.set_text(f'Points: {curve._N}')
        
        return line, time_text, points_text
    
    # アニメーション作成（200フレーム、20ms間隔）
    anim = FuncAnimation(fig, update, frames=200, interval=20, blit=True)
    
    plt.title('Moving Curve Simulation (step_W_uniform)')
    plt.show()
