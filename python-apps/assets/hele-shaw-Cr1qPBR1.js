import"./modulepreload-polyfill-B5Qt9EMX.js";import{r as a,j as e,c as b}from"./vendor-three-DWdYVK8p.js";import{u as w}from"./usePyodide-Bpngy5wJ.js";import"./vendor-react-CCYiWZgt.js";const j="/my-site/python-apps/";function S(){const p=a.useRef(null),[s,m]=a.useState(!1),l=a.useRef(!1),t=a.useRef(null),c=a.useRef(!1),{pyodide:i,isReady:r,error:_}=w(["moving_curve_new.py","mfs_hele_shaw.py"],{packages:["matplotlib"]}),u=a.useCallback(async()=>{if(!i||!r||!p.current)return;const y=await(await fetch(`${j}/pi_contour.npy`)).arrayBuffer(),d=new Uint8Array(y);i.FS.writeFile("/pi_contour.npy",d);const o=document.querySelector('div[id^="matplotlib_"]');o&&o.remove(),p.current.innerHTML='<div class="loading-placeholder"><p>Initializing simulation...</p></div>',await i.runPythonAsync(`
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
`),c.current=!0},[i,r]),f=a.useCallback(async()=>{!i||!r||!c.current||await i.runPythonAsync(`
y_pts, z_pts, Q_hat = step_simulation(curve)
simulation_step += 1
update_plot()
fig.canvas.draw()
`)},[i,r]);a.useEffect(()=>{if(l.current=s,!s){t.current&&(cancelAnimationFrame(t.current),t.current=null);return}let n=0;const d=1e3/10,o=async v=>{l.current&&(v-n>=d&&(await f(),n=v),l.current&&(t.current=requestAnimationFrame(o)))};return t.current=requestAnimationFrame(o),()=>{t.current&&(cancelAnimationFrame(t.current),t.current=null)}},[s,f]),a.useEffect(()=>{r&&!c.current&&u()},[r,u]);const x=()=>m(!0),h=()=>{m(!1),l.current=!1,t.current&&(cancelAnimationFrame(t.current),t.current=null)},g=async()=>{h(),c.current=!1;const n=document.querySelector('div[id^="matplotlib_"]');n&&n.remove(),await u()};return e.jsxs("div",{className:"app-container",children:[e.jsxs("header",{className:"app-header",children:[e.jsx("h1",{children:"🌊 Hele-Shaw Flow Simulation"}),e.jsx("p",{className:"subtitle",children:"基本解近似解法（MFS）によるヘレショウ流れ"})]}),_&&e.jsxs("div",{className:"error-message",children:["Error: ",_]}),e.jsx("div",{className:"figure-wrapper",ref:p,children:!r&&e.jsx("div",{className:"loading-placeholder",children:e.jsx("p",{children:"Loading Python & Matplotlib..."})})}),e.jsxs("div",{className:"controls",children:[e.jsxs("div",{className:"buttons",children:[s?e.jsx("button",{className:"btn btn-warning",onClick:h,children:"⏸ 停止"}):e.jsx("button",{className:"btn btn-success",onClick:x,disabled:!r,children:"▶ 再生"}),e.jsx("button",{className:"btn btn-secondary",onClick:g,disabled:!r,children:"🔄 Restart"})]}),!r&&e.jsx("p",{className:"loading-text",children:"Loading Python environment..."})]}),e.jsx("footer",{className:"app-footer",children:e.jsx("p",{children:e.jsx("a",{href:"/my-site/python-apps/",children:"← Back to Apps"})})})]})}b.createRoot(document.getElementById("root")).render(e.jsx(a.StrictMode,{children:e.jsx(S,{})}));
