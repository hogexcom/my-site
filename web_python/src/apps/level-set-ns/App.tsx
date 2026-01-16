import { useCallback, useEffect, useRef, useState } from 'react'
import { usePyodide } from '../../hooks/usePyodide'
import './App.css'

const BASE_URL = import.meta.env.BASE_URL

const defaultConfig = {
  resolution: 40,
  dt: 0.002,
  re: 1000,
  stepsPerFrame: 2,
  initialCondition: 'column'
}

type Config = typeof defaultConfig

function App() {
  const figureRef = useRef<HTMLDivElement>(null)
  const [config, setConfig] = useState<Config>(defaultConfig)
  const [isPlaying, setIsPlaying] = useState(false)
  const [stepCount, setStepCount] = useState(0)
  const [isInitializing, setIsInitializing] = useState(false)
  const [simError, setSimError] = useState<string | null>(null)
  const configRef = useRef<Config>(defaultConfig)
  const skipRestartRef = useRef(true)
  const isPlayingRef = useRef(false)
  const animationRef = useRef<number | null>(null)
  const isInitializedRef = useRef(false)

  const { pyodide, isReady, error: pyodideError } = usePyodide(
    [
      'level_set_method/LevelSetFlowSolver.py',
      'level_set_method/ParticleLevelSetSolver.py',
      'level_set_method/WENO2D.py',
      'level_set_method/WENO.py'
    ],
    { packages: ['matplotlib'] }
  )

  const initSimulation = useCallback(async () => {
    if (!pyodide || !isReady || !figureRef.current) return

    setStepCount(0)
    isInitializedRef.current = false
    setIsInitializing(true)
    setSimError(null)

    document.querySelectorAll('div[id^="matplotlib_"]').forEach((node) => node.remove())
    figureRef.current.innerHTML = ''

    try {
      await pyodide.runPythonAsync(`
import numpy as np
import js

import matplotlib
matplotlib.use('module://matplotlib_pyodide.wasm_backend')
import matplotlib.pyplot as plt

from ParticleLevelSetSolver import ParticleLevelSetSolver

nx = ${config.resolution}
ny = ${config.resolution}
re = ${config.re}
dt = ${config.dt}
ui_initial_condition = ${JSON.stringify(config.initialCondition)}
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
`)

      isInitializedRef.current = true
    } catch (error) {
      console.error('Failed to initialize simulation:', error)
      setSimError(error instanceof Error ? error.message : String(error))
    } finally {
      setIsInitializing(false)
    }
  }, [pyodide, isReady, config])

  const stepSimulation = useCallback(
    async (steps: number) => {
      if (!pyodide || !isReady || !isInitializedRef.current) return

      try {
        await pyodide.runPythonAsync(`
for _ in range(${steps}):
    if pouring_enabled:
        apply_source()
    solver.solve_ns_step()
    step_count += 1
update_plot()
fig.canvas.draw()
`)

        setStepCount((prev) => prev + steps)
      } catch (error) {
        console.error('Simulation step failed:', error)
        setSimError(error instanceof Error ? error.message : String(error))
        setIsPlaying(false)
        isPlayingRef.current = false
      }
    },
    [pyodide, isReady]
  )

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
    const fps = 10
    const interval = 1000 / fps

    const animate = async (time: number) => {
      if (!isPlayingRef.current) return

      if (time - lastTime >= interval) {
        await stepSimulation(config.stepsPerFrame)
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
  }, [isPlaying, stepSimulation, config.stepsPerFrame])

  const handlePlay = () => setIsPlaying(true)
  const handlePause = () => {
    setIsPlaying(false)
    isPlayingRef.current = false
  }
  const handleStep = async () => {
    await stepSimulation(1)
  }
  const handleRestart = useCallback(async () => {
    setIsPlaying(false)
    isPlayingRef.current = false
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
      animationRef.current = null
    }
    await initSimulation()
  }, [initSimulation])

  useEffect(() => {
    if (isReady && !isInitializedRef.current) {
      initSimulation()
    }
  }, [isReady, initSimulation])

  useEffect(() => {
    if (!isReady) return
    if (skipRestartRef.current) {
      skipRestartRef.current = false
      configRef.current = config
      return
    }
    if (
      configRef.current.initialCondition === config.initialCondition &&
      configRef.current.resolution === config.resolution &&
      configRef.current.dt === config.dt
    ) {
      return
    }
    configRef.current = config
    if (isInitializedRef.current) {
      handleRestart()
    }
  }, [config, isReady, handleRestart])

  const controlsDisabled = !isReady || isInitializing
  const isPouring = config.initialCondition === 'pouring'
  const configDisabled = controlsDisabled || isPlaying

  const statusLabel = pyodideError || simError
    ? 'Simulation error'
    : !isReady
      ? 'Loading Pyodide...'
      : isInitializing
        ? 'Initializing simulation...'
        : isPlaying
          ? `Running | Step ${stepCount}`
          : `Ready | Step ${stepCount}`

  return (
    <div className="ls-container">
      <header className="ls-header">
        <h1 className="ls-title">Level Set + Navier-Stokes (2D)</h1>
        <p className="ls-subtitle">
          Level set advection (WENO5 + TVD-RK3) and incompressible flow coupling executed in Pyodide.
          Adjust the grid and timestep to balance stability and performance.
        </p>
      </header>

      <section className="ls-layout">
        <div className="ls-panel">
          <h2>Simulation Controls</h2>

          <div className="ls-control">
            <label htmlFor="initial-condition">Initial condition</label>
            <select
              id="initial-condition"
              value={config.initialCondition}
              disabled={configDisabled}
              onChange={(event) => {
                const nextCondition = event.target.value
                setConfig((prev) => {
                  const nextConfig = { ...prev, initialCondition: nextCondition }
                  if (nextCondition === 'pouring') {
                    nextConfig.resolution = Math.max(nextConfig.resolution, 100)
                    nextConfig.dt = 0.001
                  }
                  return nextConfig
                })
              }}
            >
              <option value="column">Dam break column</option>
              {/* <option value="flat">Flat surface</option>
              <option value="empty">Empty (air only)</option> */}
              <option value="pouring">Pouring from top</option>
            </select>
          </div>

          <div className="ls-control">
            <label htmlFor="resolution">Grid resolution</label>
            <select
              id="resolution"
              value={config.resolution}
              disabled={configDisabled}
              onChange={(event) => {
                const nextResolution = Number(event.target.value)
                setConfig((prev) => ({
                  ...prev,
                  resolution: prev.initialCondition === 'pouring'
                    ? Math.max(nextResolution, 70)
                    : nextResolution
                }))
              }}
            >
              <option value={30}>30 x 30 (fast)</option>
              <option value={40}>40 x 40</option>
              <option value={50}>50 x 50</option>
              <option value={60}>60 x 60</option>
              <option value={70}>70 x 70</option>
              <option value={80}>80 x 80</option>
              <option value={90}>90 x 90</option>
              <option value={100}>100 x 100</option>
            </select>
          </div>

          <div className="ls-control">
            <label htmlFor="dt">Time step</label>
            <input
              id="dt"
              type="number"
              min="0.0005"
              max="0.01"
              step="0.0005"
              value={config.dt}
              disabled={configDisabled || isPouring}
              onChange={(event) => {
                const nextDt = Number(event.target.value)
                setConfig((prev) => ({
                  ...prev,
                  dt: prev.initialCondition === 'pouring' ? 0.001 : nextDt
                }))
              }}
            />
          </div>

          <div className="ls-control">
            <label htmlFor="steps">Steps per frame</label>
            <input
              id="steps"
              type="number"
              min="1"
              max="6"
              step="1"
              value={config.stepsPerFrame}
              disabled={controlsDisabled}
              onChange={(event) =>
                setConfig((prev) => ({ ...prev, stepsPerFrame: Number(event.target.value) }))
              }
            />
          </div>

          <div className="ls-buttons">
            {!isPlaying ? (
              <button className="ls-button" onClick={handlePlay} disabled={controlsDisabled}>
                Start
              </button>
            ) : (
              <button className="ls-button" onClick={handlePause}>
                Pause
              </button>
            )}
            <button
              className="ls-button secondary"
              onClick={handleStep}
              disabled={controlsDisabled || isPlaying}
            >
              Step
            </button>
            <button className="ls-button secondary" onClick={handleRestart} disabled={controlsDisabled}>
              Restart
            </button>
          </div>

          <p className="ls-warning">
            Higher resolution and smaller time steps improve stability but can slow down Pyodide.
            Pouring mode forces &gt;=70 x 70 grid and dt=0.001.
          </p>

          {pyodideError && <p className="ls-warning">Pyodide error: {pyodideError}</p>}
          {simError && <p className="ls-warning">Simulation error: {simError}</p>}

          <p className="ls-legend">
            Blue region: liquid (phi &lt; 0). The black contour marks the free surface.
          </p>

          <p className="ls-legend">
            <a href={`${BASE_URL}`}>← Back to Apps</a>
          </p>
        </div>

        <div className="ls-canvas-wrapper">
          <div className="ls-status">{statusLabel}</div>
          <div className="ls-canvas" ref={figureRef}>
            {!isReady && <div className="ls-status">Loading Python &amp; Matplotlib...</div>}
          </div>
        </div>
      </section>
    </div>
  )
}

export default App
