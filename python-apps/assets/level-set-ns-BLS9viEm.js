import"./modulepreload-polyfill-B5Qt9EMX.js";import{r as n,j as e,c as q}from"./vendor-three-DWdYVK8p.js";import{u as z}from"./usePyodide-Bpngy5wJ.js";import"./vendor-react-CCYiWZgt.js";const I="/my-site/python-apps/",F={resolution:40,dt:.002,re:1e3,stepsPerFrame:2,initialCondition:"column"};function D(){const h=n.useRef(null),[i,_]=n.useState(F),[o,x]=n.useState(!1),[C,P]=n.useState(0),[R,k]=n.useState(!1),[y,b]=n.useState(null),p=n.useRef(F),w=n.useRef(!0),u=n.useRef(!1),l=n.useRef(null),m=n.useRef(!1),{pyodide:d,isReady:r,error:g}=z(["level_set_method/LevelSetFlowSolver.py","level_set_method/ParticleLevelSetSolver.py","level_set_method/WENO2D.py","level_set_method/WENO.py"],{packages:["matplotlib"]}),v=n.useCallback(async()=>{if(!(!d||!r||!h.current)){P(0),m.current=!1,k(!0),b(null),document.querySelectorAll('div[id^="matplotlib_"]').forEach(t=>t.remove()),h.current.innerHTML="";try{await d.runPythonAsync(`
import numpy as np
import js

import matplotlib
matplotlib.use('module://matplotlib_pyodide.wasm_backend')
import matplotlib.pyplot as plt

from ParticleLevelSetSolver import ParticleLevelSetSolver

nx = ${i.resolution}
ny = ${i.resolution}
re = ${i.re}
dt = ${i.dt}
ui_initial_condition = ${JSON.stringify(i.initialCondition)}
initial_condition = "empty" if ui_initial_condition == "pouring" else ui_initial_condition

solver = ParticleLevelSetSolver(nx=nx, ny=ny, re=re, dt=dt, initial_condition=initial_condition)
step_count = 0
pouring_enabled = (ui_initial_condition == "pouring")

if pouring_enabled:
    solver.phi = np.minimum(solver.phi, solver.Y - 0.1)
    if hasattr(solver, "init_particles"):
        solver.init_particles()
    faucet_x_min = 0.45
    faucet_x_max = 0.55
    faucet_y_min = 0.90
    faucet_y_max = 1.00
    faucet_vel = -1.0

    def apply_source():
        X, Y = solver.X, solver.Y
        d_box = np.maximum.reduce([
            X - faucet_x_max,
            faucet_x_min - X,
            Y - faucet_y_max,
            faucet_y_min - Y
        ])
        solver.phi = np.minimum(solver.phi, d_box)
        dy = solver.dy
        dx = solver.dx
        y_v = np.linspace(0, solver.ly, ny + 1)
        x_v = np.linspace(dx / 2, solver.lx - dx / 2, nx)
        X_v, Y_v = np.meshgrid(x_v, y_v)
        mask_v_source = (X_v >= faucet_x_min) & (X_v <= faucet_x_max) & (Y_v >= faucet_y_min)
        solver.v[mask_v_source] = faucet_vel
        x_u = np.linspace(0, solver.lx, nx + 1)
        y_u = np.linspace(dy / 2, solver.ly - dy / 2, ny)
        X_u, Y_u = np.meshgrid(x_u, y_u)
        mask_u_source = (X_u >= faucet_x_min) & (X_u <= faucet_x_max) & (Y_u >= faucet_y_min)
        solver.u[mask_u_source] = 0.0

fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor('#0b1220')

def update_plot():
    global step_count
    ax.clear()
    ax.set_facecolor('#0b1220')
    phi = solver.phi
    ax.contourf(solver.X, solver.Y, phi, levels=[-10, 0, 10], colors=['#2563eb', '#38bdf8'], alpha=0.85)
    ax.contour(solver.X, solver.Y, phi, levels=[0], colors='#0f172a', linewidths=2)
    water_area = np.sum(phi < 0) * solver.dx * solver.dy
    ax.set_title(f"Step {step_count} | Water area={water_area:.3f}", color='#e2e8f0', fontsize=12)
    ax.set_xlim(0, solver.lx)
    ax.set_ylim(0, solver.ly)
    ax.set_aspect('equal')
    ax.tick_params(colors='#cbd5f5', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#94a3b8')
    plt.tight_layout()

update_plot()
plt.show()

matplotlib_div = js.document.querySelector('div[id^="matplotlib_"]')
if matplotlib_div:
    target = js.document.querySelector('.ls-canvas')
    if target:
        target.innerHTML = ''
        target.appendChild(matplotlib_div)
    toolbar = matplotlib_div.querySelector('div:last-child')
    if toolbar:
        toolbar.style.display = 'none'
    title = matplotlib_div.querySelector('div[id$="top"]')
    if title:
        title.style.display = 'none'
`),m.current=!0}catch(t){console.error("Failed to initialize simulation:",t),b(t instanceof Error?t.message:String(t))}finally{k(!1)}}},[d,r,i]),j=n.useCallback(async t=>{if(!(!d||!r||!m.current))try{await d.runPythonAsync(`
for _ in range(${t}):
    if pouring_enabled:
        apply_source()
    solver.solve_ns_step()
    step_count += 1
update_plot()
fig.canvas.draw()
`),P(s=>s+t)}catch(s){console.error("Simulation step failed:",s),b(s instanceof Error?s.message:String(s)),x(!1),u.current=!1}},[d,r]);n.useEffect(()=>{if(u.current=o,!o){l.current&&(cancelAnimationFrame(l.current),l.current=null);return}let t=0;const a=1e3/10,c=async E=>{u.current&&(E-t>=a&&(await j(i.stepsPerFrame),t=E),u.current&&(l.current=requestAnimationFrame(c)))};return l.current=requestAnimationFrame(c),()=>{l.current&&(cancelAnimationFrame(l.current),l.current=null)}},[o,j,i.stepsPerFrame]);const L=()=>x(!0),A=()=>{x(!1),u.current=!1},X=async()=>{await j(1)},S=n.useCallback(async()=>{x(!1),u.current=!1,l.current&&(cancelAnimationFrame(l.current),l.current=null),await v()},[v]);n.useEffect(()=>{r&&!m.current&&v()},[r,v]),n.useEffect(()=>{if(r){if(w.current){w.current=!1,p.current=i;return}p.current.initialCondition===i.initialCondition&&p.current.resolution===i.resolution&&p.current.dt===i.dt||(p.current=i,m.current&&S())}},[i,r,S]);const f=!r||R,Y=i.initialCondition==="pouring",N=f||o,$=g||y?"Simulation error":r?R?"Initializing simulation...":o?`Running | Step ${C}`:`Ready | Step ${C}`:"Loading Pyodide...";return e.jsxs("div",{className:"ls-container",children:[e.jsxs("header",{className:"ls-header",children:[e.jsx("h1",{className:"ls-title",children:"Level Set + Navier-Stokes (2D)"}),e.jsx("p",{className:"ls-subtitle",children:"Level set advection (WENO5 + TVD-RK3) and incompressible flow coupling executed in Pyodide. Adjust the grid and timestep to balance stability and performance."})]}),e.jsxs("section",{className:"ls-layout",children:[e.jsxs("div",{className:"ls-panel",children:[e.jsx("h2",{children:"Simulation Controls"}),e.jsxs("div",{className:"ls-control",children:[e.jsx("label",{htmlFor:"initial-condition",children:"Initial condition"}),e.jsxs("select",{id:"initial-condition",value:i.initialCondition,disabled:N,onChange:t=>{const s=t.target.value;_(a=>{const c={...a,initialCondition:s};return s==="pouring"&&(c.resolution=Math.max(c.resolution,100),c.dt=.001),c})},children:[e.jsx("option",{value:"column",children:"Dam break column"}),e.jsx("option",{value:"pouring",children:"Pouring from top"})]})]}),e.jsxs("div",{className:"ls-control",children:[e.jsx("label",{htmlFor:"resolution",children:"Grid resolution"}),e.jsxs("select",{id:"resolution",value:i.resolution,disabled:N,onChange:t=>{const s=Number(t.target.value);_(a=>({...a,resolution:a.initialCondition==="pouring"?Math.max(s,70):s}))},children:[e.jsx("option",{value:30,children:"30 x 30 (fast)"}),e.jsx("option",{value:40,children:"40 x 40"}),e.jsx("option",{value:50,children:"50 x 50"}),e.jsx("option",{value:60,children:"60 x 60"}),e.jsx("option",{value:70,children:"70 x 70"}),e.jsx("option",{value:80,children:"80 x 80"}),e.jsx("option",{value:90,children:"90 x 90"}),e.jsx("option",{value:100,children:"100 x 100"})]})]}),e.jsxs("div",{className:"ls-control",children:[e.jsx("label",{htmlFor:"dt",children:"Time step"}),e.jsx("input",{id:"dt",type:"number",min:"0.0005",max:"0.01",step:"0.0005",value:i.dt,disabled:N||Y,onChange:t=>{const s=Number(t.target.value);_(a=>({...a,dt:a.initialCondition==="pouring"?.001:s}))}})]}),e.jsxs("div",{className:"ls-control",children:[e.jsx("label",{htmlFor:"steps",children:"Steps per frame"}),e.jsx("input",{id:"steps",type:"number",min:"1",max:"6",step:"1",value:i.stepsPerFrame,disabled:f,onChange:t=>_(s=>({...s,stepsPerFrame:Number(t.target.value)}))})]}),e.jsxs("div",{className:"ls-buttons",children:[o?e.jsx("button",{className:"ls-button",onClick:A,children:"Pause"}):e.jsx("button",{className:"ls-button",onClick:L,disabled:f,children:"Start"}),e.jsx("button",{className:"ls-button secondary",onClick:X,disabled:f||o,children:"Step"}),e.jsx("button",{className:"ls-button secondary",onClick:S,disabled:f,children:"Restart"})]}),e.jsx("p",{className:"ls-warning",children:"Higher resolution and smaller time steps improve stability but can slow down Pyodide. Pouring mode forces >=70 x 70 grid and dt=0.001."}),g&&e.jsxs("p",{className:"ls-warning",children:["Pyodide error: ",g]}),y&&e.jsxs("p",{className:"ls-warning",children:["Simulation error: ",y]}),e.jsx("p",{className:"ls-legend",children:"Blue region: liquid (phi < 0). The black contour marks the free surface."}),e.jsx("p",{className:"ls-legend",children:e.jsx("a",{href:`${I}`,children:"← Back to Apps"})})]}),e.jsxs("div",{className:"ls-canvas-wrapper",children:[e.jsx("div",{className:"ls-status",children:$}),e.jsx("div",{className:"ls-canvas",ref:h,children:!r&&e.jsx("div",{className:"ls-status",children:"Loading Python & Matplotlib..."})})]})]})]})}q.createRoot(document.getElementById("root")).render(e.jsx(n.StrictMode,{children:e.jsx(D,{})}));
