import numpy as np
import random
import cmath

def generate_fractal_1d(N, H):
    """
    1次元フラクタルノイズを生成
    
    Parameters:
    -----------
    N : int
        サンプル数（奇数推奨）
    H : float
        ハースト指数 (0 < H < 1)
        H=0.5: ホワイトノイズ
        H<0.5: 荒いノイズ
        H>0.5: 滑らかなノイズ
    
    Returns:
    --------
    ndarray
        生成されたフラクタル信号
    """
    beta = 2 * H + 1
    
    A_0 = np.zeros(1, dtype=complex)
    A_pos = np.zeros((N//2,), dtype=complex)
    
    for i in range(N//2):
        # パワースペクトル密度 P_i ∝ (i+1)^(-beta) にするため、
        # 振幅は (i+1)^(-beta/2) とする
        rad = pow(i+1, -beta / 2) * random.gauss(0, 1)
        phase = 2 * cmath.pi * random.random()
        A_pos[i] = rad * cmath.exp(phase * 1j)
    
    A_neg = np.conjugate(A_pos[::-1])
    A = np.concatenate((A_0, A_pos, A_neg))
    
    f = np.fft.ifft(A)
    
    return f.real
