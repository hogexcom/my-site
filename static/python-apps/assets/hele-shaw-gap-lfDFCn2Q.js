import"./modulepreload-polyfill-B5Qt9EMX.js";import{r as i,j as e,R as g,b}from"./vendor-three-CePzC98V.js";import{u as j}from"./usePyodide-SVK5obCv.js";import"./vendor-react-CCYiWZgt.js";function w(){const c=i.useRef(null),[n,u]=i.useState(!1),s=i.useRef(!1),t=i.useRef(null),o=i.useRef(!1),{pyodide:l,isReady:a,error:d}=j(["moving_curve_new.py","mfs_hele_shaw_gap.py"],{packages:["matplotlib"]}),p=i.useCallback(async()=>{if(!l||!a||!c.current)return;const r=document.querySelector('div[id^="matplotlib_"]');r&&r.remove(),c.current.innerHTML='<div class="loading-placeholder"><p>Initializing simulation...</p></div>',await l.runPythonAsync(`
import numpy as np
import js
from pyodide.ffi import create_proxy

from moving_curve_new import MovingCurve
from mfs_hele_shaw_gap import step_simulation, get_curve_data, contour_data, MFS, initial_data, N

# 初期曲線を作成（摂動付き円）
X = initial_data(N)
print(f"Created initial data: {X.shape}")

# 曲線を初期化
curve = MovingCurve(X, epsilon=0.3)
curve.dt = 50 * N**-2
simulation_step = 0

import matplotlib
matplotlib.use('module://matplotlib_pyodide.wasm_backend')
import matplotlib.pyplot as plt

# パラメータ
xlim = 2
ylim = 2

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
    contour_u, contour_v, contour_p = contour_data(y_pts, z_pts, Q_hat, xlim=xlim, ylim=ylim, resolution=30)
    
    # 等高線を描画
    U = np.array(contour_u)
    V = np.array(contour_v)
    P = np.array(contour_p)
    
    # 圧力の等高線
    levels = np.linspace(np.nanmin(P), np.nanmax(P), 15)
    cs = ax.contour(U, V, P, levels=levels, colors='limegreen', alpha=0.6, linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f', colors='white')
    
    # 曲線を描画
    ax.plot(curve_x, curve_y, '-', color='#00BFFF', linewidth=2, label='Curve')
    ax.plot(curve_x[:-1], curve_y[:-1], 'o', color='orange', markersize=3)
    
    # 設定
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, color='#555')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.set_title(f'Step: {simulation_step}, t={curve.elapsed_time:.4f}', color='white', fontsize=14)
    
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
`),o.current=!0},[l,a]),m=i.useCallback(async()=>{!l||!a||!o.current||await l.runPythonAsync(`
y_pts, z_pts, Q_hat = step_simulation(curve)
simulation_step += 1
update_plot()
fig.canvas.draw()
`)},[l,a]);i.useEffect(()=>{if(s.current=n,!n){t.current&&(cancelAnimationFrame(t.current),t.current=null);return}let r=0;const v=1e3/10,f=async h=>{s.current&&(h-r>=v&&(await m(),r=h),s.current&&(t.current=requestAnimationFrame(f)))};return t.current=requestAnimationFrame(f),()=>{t.current&&(cancelAnimationFrame(t.current),t.current=null)}},[n,m]),i.useEffect(()=>{a&&!o.current&&p()},[a,p]);const x=()=>u(!0),_=()=>{u(!1),s.current=!1,t.current&&(cancelAnimationFrame(t.current),t.current=null)},y=async()=>{_(),o.current=!1;const r=document.querySelector('div[id^="matplotlib_"]');r&&r.remove(),await p()};return e.jsxs("div",{className:"app-container",children:[e.jsxs("header",{className:"app-header",children:[e.jsx("h1",{children:"🫧 Hele-Shaw Gap Rising Flow"}),e.jsx("p",{className:"subtitle",children:"基本解近似解法（MFS）による隙間上昇ヘレショウ流れ"})]}),d&&e.jsxs("div",{className:"error-message",children:["Error: ",d]}),e.jsx("div",{className:"figure-wrapper",ref:c,children:!a&&e.jsx("div",{className:"loading-placeholder",children:e.jsx("p",{children:"Loading Python & Matplotlib..."})})}),e.jsxs("div",{className:"controls",children:[e.jsxs("div",{className:"buttons",children:[n?e.jsx("button",{className:"btn btn-warning",onClick:_,children:"⏸ 停止"}):e.jsx("button",{className:"btn btn-success",onClick:x,disabled:!a,children:"▶ 再生"}),e.jsx("button",{className:"btn btn-secondary",onClick:y,disabled:!a,children:"🔄 Restart"})]}),!a&&e.jsx("p",{className:"loading-text",children:"Loading Python environment..."})]}),e.jsx("footer",{className:"app-footer",children:e.jsx("p",{children:e.jsx("a",{href:"/my-site/python-apps/",children:"← Back to Apps"})})})]})}g.createRoot(document.getElementById("root")).render(e.jsx(b.StrictMode,{children:e.jsx(w,{})}));
