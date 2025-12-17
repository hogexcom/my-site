import numpy as np
import random
import cmath

def generate_fractal_2d(N, H):
    """
    2次元フラクタルノイズを生成
    
    Parameters:
    -----------
    N : int
        解像度（NxNの配列を生成、奇数推奨）
    H : float
        ハースト指数 (0 < H < 1)
        H=0.5: ホワイトノイズ
        H<0.5: 荒いノイズ
        H>0.5: 滑らかなノイズ
    
    Returns:
    --------
    ndarray
        生成されたフラクタル高さマップ（NxN配列）
    """
    beta = H + 1
    
    A_00 = np.zeros(1, dtype=complex)
    A_01 = np.zeros((N // 2,), dtype=complex)
    A_10 = np.zeros((N // 2, 1), dtype=complex)
    A_11 = np.zeros((N // 2, N // 2), dtype=complex)
    A_12 = np.zeros((N // 2, N // 2), dtype=complex)
    
    # 水平周波数成分
    for u in range(N // 2):
        rad = pow((u + 1) ** 2, -beta / 2) * random.gauss(0, 1)
        phase = 2 * cmath.pi * random.random()
        A_01[u] = rad * cmath.exp(phase * 1j)
    
    A_02 = np.conjugate(A_01[::-1])
    
    # 垂直周波数成分
    for v in range(N // 2):
        rad = pow((v + 1) ** 2, -beta / 2) * random.gauss(0, 1)
        phase = 2 * cmath.pi * random.random()
        A_10[v, 0] = rad * cmath.exp(phase * 1j)
    
    A_20 = np.conjugate(A_10[::-1])
    
    # 第1象限の周波数成分
    for u in range(N // 2):
        for v in range(N // 2):
            rad = pow((u + 1) ** 2 + (v + 1) ** 2, -beta / 2) * random.gauss(0, 1)
            phase = 2 * cmath.pi * random.random()
            A_11[u, v] = rad * cmath.exp(phase * 1j)
    
    A_22 = np.conjugate(np.flip(A_11))
    
    # 第2象限の周波数成分
    for u in range(N // 2):
        for v in range(N // 2):
            rad = pow((u + 1) ** 2 + (v + 1) ** 2, -beta / 2) * random.gauss(0, 1)
            phase = 2 * cmath.pi * random.random()
            A_12[u, v] = rad * cmath.exp(phase * 1j)
    
    A_12 = np.fliplr(A_12)
    A_21 = np.conjugate(np.flip(A_12))
    
    # フーリエ係数配列を構成
    A = np.block([[A_00, A_01, A_02], [A_10, A_11, A_12], [A_20, A_21, A_22]])
    
    # 逆フーリエ変換
    f = np.fft.ifft2(A)
    
    return f.real
