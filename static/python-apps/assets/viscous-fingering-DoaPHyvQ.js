import"./modulepreload-polyfill-B5Qt9EMX.js";import{r,j as e,c as y}from"./vendor-three-CePzC98V.js";import{u as b}from"./usePyodide-SVK5obCv.js";import"./vendor-react-CCYiWZgt.js";function j(){const c=r.useRef(null),[n,p]=r.useState(!1),l=r.useRef(!1),t=r.useRef(null),o=r.useRef(!1),{pyodide:s,isReady:i,error:d}=b(["moving_curve_new.py","mfs_hele_shaw_viscous_fingering.py"],{packages:["matplotlib"]}),u=r.useCallback(async()=>{if(!s||!i||!c.current)return;const a=document.querySelector('div[id^="matplotlib_"]');a&&a.remove(),c.current.innerHTML='<div class="loading-placeholder"><p>Initializing simulation...</p></div>',await s.runPythonAsync(`
import numpy as np
import js

from moving_curve_new import MovingCurve
import mfs_hele_shaw_viscous_fingering as vf

# 初期曲線を作成
X = vf.initial_data(vf.N, reverse=True)
C = np.sum(X, axis=0) / vf.N
X = X - C

curve = MovingCurve(X, epsilon=0.3)
curve.dt = 5e-4

simulation_step = 0

import matplotlib
matplotlib.use('module://matplotlib_pyodide.wasm_backend')
import matplotlib.pyplot as plt

# Figure作成
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')

def update_plot():
    global simulation_step
    ax.clear()
    ax.set_facecolor('#1a1a1a')
    
    curve_x, curve_y = vf.get_curve_data(curve)
    ax.plot(curve_x, curve_y, '.-', color='#00BFFF', linewidth=2, markersize=3, markerfacecolor='#ff7f00', markeredgecolor='#ff7f00')
    
    ax.set_xlim(-vf.xlim, vf.xlim)
    ax.set_ylim(-vf.ylim, vf.ylim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, color='#555')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.set_title(f'Viscous Fingering - Step: {simulation_step}', color='white', fontsize=14)
    
    plt.tight_layout()

update_plot()
plt.show()

# matplotlibが生成したdivを正しい位置に移動
matplotlib_div = js.document.querySelector('div[id^="matplotlib_"]')
if matplotlib_div:
    target = js.document.querySelector('.figure-wrapper')
    if target:
        target.innerHTML = ''
        target.appendChild(matplotlib_div)
    toolbar = matplotlib_div.querySelector('div:last-child')
    if toolbar:
        toolbar.style.display = 'none'
    title = matplotlib_div.querySelector('div[id$="top"]')
    if title:
        title.style.display = 'none'
`),o.current=!0},[s,i]),m=r.useCallback(async()=>{!s||!i||!o.current||await s.runPythonAsync(`
vf.step_simulation(curve)
simulation_step += 1
update_plot()
fig.canvas.draw()
`)},[s,i]);r.useEffect(()=>{if(l.current=n,!n){t.current&&(cancelAnimationFrame(t.current),t.current=null);return}let a=0;const g=1e3/10,v=async h=>{l.current&&(h-a>=g&&(await m(),a=h),l.current&&(t.current=requestAnimationFrame(v)))};return t.current=requestAnimationFrame(v),()=>{t.current&&(cancelAnimationFrame(t.current),t.current=null)}},[n,m]),r.useEffect(()=>{i&&!o.current&&u()},[i,u]);const _=()=>p(!0),f=()=>{p(!1),l.current=!1,t.current&&(cancelAnimationFrame(t.current),t.current=null)},x=async()=>{f(),o.current=!1;const a=document.querySelector('div[id^="matplotlib_"]');a&&a.remove(),await u()};return e.jsxs("div",{className:"app-container",children:[e.jsxs("header",{className:"app-header",children:[e.jsx("h1",{children:"🖐️ Viscous Fingering Simulation"}),e.jsx("p",{className:"subtitle",children:"Saffman-Taylor不安定性による指状パターン形成"})]}),d&&e.jsxs("div",{className:"error-message",children:["Error: ",d]}),e.jsx("div",{className:"figure-wrapper",ref:c,children:!i&&e.jsx("div",{className:"loading-placeholder",children:e.jsx("p",{children:"Loading Python & Matplotlib..."})})}),e.jsxs("div",{className:"controls",children:[e.jsxs("div",{className:"buttons",children:[n?e.jsx("button",{className:"btn btn-warning",onClick:f,children:"⏸ 停止"}):e.jsx("button",{className:"btn btn-success",onClick:_,disabled:!i,children:"▶ 再生"}),e.jsx("button",{className:"btn btn-secondary",onClick:x,disabled:!i,children:"🔄 Restart"})]}),!i&&e.jsx("p",{className:"loading-text",children:"Loading Python environment..."})]}),e.jsx("footer",{className:"app-footer",children:e.jsx("p",{children:e.jsx("a",{href:"/my-site/python-apps/",children:"← Back to Apps"})})})]})}y.createRoot(document.getElementById("root")).render(e.jsx(r.StrictMode,{children:e.jsx(j,{})}));
