import { useState, useRef, useCallback, useEffect } from 'react'
import { usePyodide } from '../../hooks/usePyodide'
import './App.css'
const BASE_URL = import.meta.env.BASE_URL


function App() {
  const figureRef = useRef<HTMLDivElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const isPlayingRef = useRef(false)
  const animationRef = useRef<number | null>(null)
  const isInitializedRef = useRef(false)
  
  const { pyodide, isReady, error: pyodideError } = usePyodide([
    'moving_curve_new.py',
    'mfs_hele_shaw.py'
  ], { packages: ['matplotlib'] })

  // シミュレーション初期化
  const initSimulation = useCallback(async () => {
    if (!pyodide || !isReady || !figureRef.current) return
    
    // πの輪郭データを読み込み
    const response = await fetch(`${BASE_URL}/pi_contour.npy`)
    const arrayBuffer = await response.arrayBuffer()
    const uint8Array = new Uint8Array(arrayBuffer)
    pyodide.FS.writeFile('/pi_contour.npy', uint8Array)
    
    // 既存のmatplotlib divを削除
    const existingDiv = document.querySelector('div[id^="matplotlib_"]')
    if (existingDiv) {
      existingDiv.remove()
    }
    
    // 既存の内容をクリア
    figureRef.current.innerHTML = '<div class="loading-placeholder"><p>Initializing simulation...</p></div>'
    
    await pyodide.runPythonAsync(`
import numpy as np
import js
from pyodide.ffi import create_proxy

from moving_curve_new import MovingCurve
from mfs_hele_shaw import step_simulation, get_curve_data, contour_data, MFS

# πの輪郭データを読み込み
X = np.load('/pi_contour.npy')
print(f"Loaded contour data: {X.shape}")

# 初期曲線を作成
curve = MovingCurve(X, epsilon=0.3)
curve.dt = 0.002
simulation_step = 0

import matplotlib
matplotlib.use('module://matplotlib_pyodide.wasm_backend')
import matplotlib.pyplot as plt

# Figure作成
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')

# 初期プロット
def update_plot():
    global simulation_step
    ax.clear()
    ax.set_facecolor('#1a1a1a')
    
    # 曲線データを取得
    curve_x, curve_y = get_curve_data(curve)
    
    # MFSで圧力場を計算
    x_pts, y_pts, z_pts, Q_hat, H_ji, n, phi = MFS(curve)
    
    # 等高線データを計算
    contour_u, contour_v, contour_p = contour_data(y_pts, z_pts, Q_hat, xlim=7, ylim=7, resolution=50)
    
    # 等高線を描画
    U = np.array(contour_u)
    V = np.array(contour_v)
    P = np.array(contour_p)
    
    # 圧力の等高線
    levels = np.linspace(np.nanmin(P), np.nanmax(P), 20)
    ax.contour(U, V, P, levels=levels, colors='limegreen', alpha=0.6, linewidths=0.8)
    
    # 曲線を描画
    ax.plot(curve_x, curve_y, '-', color='#00BFFF', linewidth=2, label='Curve')
    ax.plot(curve_x[:-1], curve_y[:-1], 'o', color='orange', markersize=4)
    
    # 設定
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, color='#555')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.set_title(f'Step: {simulation_step}', color='white', fontsize=14)
    
    plt.tight_layout()

update_plot()
plt.show()

# matplotlibが生成したdivを正しい位置に移動
matplotlib_div = js.document.querySelector('div[id^="matplotlib_"]')
if matplotlib_div:
    target = js.document.querySelector('.figure-wrapper')
    if target:
        target.innerHTML = ''  # 既存のloading placeholderを削除
        target.appendChild(matplotlib_div)
    # ツールバーとタイトルを非表示
    toolbar = matplotlib_div.querySelector('div:last-child')
    if toolbar:
        toolbar.style.display = 'none'
    title = matplotlib_div.querySelector('div[id$="top"]')
    if title:
        title.style.display = 'none'
`)
    
    isInitializedRef.current = true
  }, [pyodide, isReady])

  // 1ステップ進める
  const stepSimulation = useCallback(async () => {
    if (!pyodide || !isReady || !isInitializedRef.current) return
    
    await pyodide.runPythonAsync(`
y_pts, z_pts, Q_hat = step_simulation(curve)
simulation_step += 1
update_plot()
fig.canvas.draw()
`)
  }, [pyodide, isReady])

  // アニメーションループ
  useEffect(() => {
    isPlayingRef.current = isPlaying
    
    if (!isPlaying) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
        animationRef.current = null
      }
      return
    }
    
    let lastTime = 0
    const fps = 10  // MFSの計算が重いのでfpsを下げる
    const interval = 1000 / fps
    
    const animate = async (time: number) => {
      if (!isPlayingRef.current) return
      
      if (time - lastTime >= interval) {
        await stepSimulation()
        lastTime = time
      }
      if (isPlayingRef.current) {
        animationRef.current = requestAnimationFrame(animate)
      }
    }
    
    animationRef.current = requestAnimationFrame(animate)
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
        animationRef.current = null
      }
    }
  }, [isPlaying, stepSimulation])

  // 初期化
  useEffect(() => {
    if (isReady && !isInitializedRef.current) {
      initSimulation()
    }
  }, [isReady, initSimulation])

  const handlePlay = () => setIsPlaying(true)
  const handleStop = () => {
    setIsPlaying(false)
    isPlayingRef.current = false
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
      animationRef.current = null
    }
  }
  const handleRestart = async () => {
    handleStop()
    isInitializedRef.current = false
    // 既存のmatplotlib divを削除
    const existingDiv = document.querySelector('div[id^="matplotlib_"]')
    if (existingDiv) {
      existingDiv.remove()
    }
    await initSimulation()
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🌊 Hele-Shaw Flow Simulation</h1>
        <p className="subtitle">基本解近似解法（MFS）によるヘレショウ流れ</p>
      </header>

      {pyodideError && (
        <div className="error-message">
          Error: {pyodideError}
        </div>
      )}

      <div className="figure-wrapper" ref={figureRef}>
        {!isReady && (
          <div className="loading-placeholder">
            <p>Loading Python & Matplotlib...</p>
          </div>
        )}
      </div>

      <div className="controls">
        <div className="buttons">
          {!isPlaying ? (
            <button 
              className="btn btn-success" 
              onClick={handlePlay}
              disabled={!isReady}
            >
              ▶ 再生
            </button>
          ) : (
            <button 
              className="btn btn-warning" 
              onClick={handleStop}
            >
              ⏸ 停止
            </button>
          )}
          
          <button 
            className="btn btn-secondary" 
            onClick={handleRestart}
            disabled={!isReady}
          >
            🔄 Restart
          </button>
        </div>
        
        {!isReady && (
          <p className="loading-text">Loading Python environment...</p>
        )}
      </div>

      <footer className="app-footer">
        <p>
          <a href={import.meta.env.BASE_URL}>← Back to Apps</a>
        </p>
      </footer>
    </div>
  )
}

export default App
