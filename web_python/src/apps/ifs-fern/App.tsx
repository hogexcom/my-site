import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { usePyodide } from '../../hooks/usePyodide'
import LinearMapVisualizer, { type LinearMapValue } from '../../components/LinearMapVisualizer'
import './App.css'

type FernPoints = {
  xs: number[]
  ys: number[]
}

type Transform = {
  id: string
  label: string
  a: number
  b: number
  c: number
  d: number
  e: number
  f: number
  p: number
}

const DEFAULT_POINTS = 20_000
const DEFAULT_BURN_IN = 200
const DEFAULT_TRANSFORMS: Transform[] = [
  { id: 'stem', label: 'Stem', a: 0.0, b: 0.0, c: 0.0, d: 0.16, e: 0.0, f: 0.0, p: 0.01 },
  { id: 'main', label: 'Main', a: 0.85, b: 0.04, c: -0.04, d: 0.85, e: 0.0, f: 1.6, p: 0.85 },
  { id: 'left', label: 'Left Leaf', a: 0.2, b: -0.26, c: 0.23, d: 0.22, e: 0.0, f: 1.6, p: 0.07 },
  { id: 'right', label: 'Right Leaf', a: -0.15, b: 0.28, c: 0.26, d: 0.24, e: 0.0, f: 0.44, p: 0.07 },
]

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function App() {
  const [pointCount, setPointCount] = useState(DEFAULT_POINTS)
  const [burnIn, setBurnIn] = useState(DEFAULT_BURN_IN)
  const [seed, setSeed] = useState(42)
  const [isGenerating, setIsGenerating] = useState(false)
  const [transforms, setTransforms] = useState<Transform[]>(DEFAULT_TRANSFORMS)
  const [selectedTransformIndex, setSelectedTransformIndex] = useState(1)
  const generationTimeoutRef = useRef<number | null>(null)
  const pointsRef = useRef<FernPoints | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const { pyodide, isReady, error: pyodideError } = usePyodide(['ifs_fern.py'])

  const drawFern = useCallback((points: FernPoints) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.round(rect.width * dpr)
    canvas.height = Math.round(rect.height * dpr)

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(1, 0, 0, 1, 0, 0)

    const width = Math.max(1, Math.floor(canvas.width))
    const height = Math.max(1, Math.floor(canvas.height))

    ctx.fillStyle = '#0b0f14'
    ctx.fillRect(0, 0, width, height)

    const { xs, ys } = points
    let minX = Infinity
    let maxX = -Infinity
    let minY = Infinity
    let maxY = -Infinity

    for (let i = 0; i < xs.length; i += 1) {
      const x = xs[i]
      const y = ys[i]
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }

    const rangeX = maxX - minX || 1
    const rangeY = maxY - minY || 1
    const scaleX = width / rangeX
    const scaleY = height / rangeY
    const scale = Math.min(scaleX, scaleY)
    const centerX = (minX + maxX) / 2
    const centerY = (minY + maxY) / 2

    const image = ctx.createImageData(width, height)
    const data = image.data

    for (let i = 0; i < xs.length; i += 1) {
      const x = xs[i]
      const y = ys[i]
      const px = Math.round((x - centerX) * scale + width / 2)
      const py = Math.round(height / 2 - (y - centerY) * scale)

      if (px < 0 || px >= width || py < 0 || py >= height) continue
      const index = (py * width + px) * 4
      data[index] = 110
      data[index + 1] = 231
      data[index + 2] = 156
      data[index + 3] = 255
    }

    ctx.putImageData(image, 0, 0)
  }, [])

  const generateFern = useCallback(async () => {
    if (!pyodide || !isReady) return
    setIsGenerating(true)

    const safePoints = clamp(pointCount, 10_000, 300_000)
    const safeBurnIn = clamp(burnIn, 1, 500)
    const safeSeed = clamp(seed, 0, 1_000_000)
    const transformsPayload = transforms.map((t) => [t.a, t.b, t.c, t.d, t.e, t.f, t.p])

    try {
      await pyodide.runPythonAsync(`
import ifs_fern
transforms = ${JSON.stringify(transformsPayload)}
xs, ys = ifs_fern.barnsley_fern(
    n=${safePoints},
    burn_in=${safeBurnIn},
    seed=${safeSeed},
    transforms=transforms,
)
xs_list = xs
ys_list = ys
      `)

      const xsList = pyodide.globals.get('xs_list').toJs()
      const ysList = pyodide.globals.get('ys_list').toJs()
      const xs = Array.from(xsList) as number[]
      const ys = Array.from(ysList) as number[]

      const points = { xs, ys }
      pointsRef.current = points
      drawFern(points)
    } catch (error) {
      console.error('IFS fern generation failed:', error)
    } finally {
      setIsGenerating(false)
    }
  }, [pyodide, isReady, pointCount, burnIn, seed, transforms, drawFern])

  useEffect(() => {
    if (!isReady) return
    if (generationTimeoutRef.current !== null) {
      window.clearTimeout(generationTimeoutRef.current)
    }
    generationTimeoutRef.current = window.setTimeout(() => {
      generateFern()
      generationTimeoutRef.current = null
    }, 250)
  }, [burnIn, generateFern, isReady, pointCount, seed, transforms])

  useEffect(() => {
    const handleResize = () => {
      if (pointsRef.current) {
        drawFern(pointsRef.current)
      }
    }
    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
      if (generationTimeoutRef.current !== null) {
        window.clearTimeout(generationTimeoutRef.current)
      }
    }
  }, [drawFern])

  const statusText = useMemo(() => {
    if (pyodideError) return `Pyodide load error: ${pyodideError}`
    if (!isReady) return 'Pyodide loading...'
    if (isGenerating) return 'Generating fern...'
    return 'Auto-updated'
  }, [isGenerating, isReady, pyodideError])

  const selectedTransform = transforms[selectedTransformIndex] ?? transforms[0]
  const selectedMapValue = useMemo<LinearMapValue>(() => {
    const transform = selectedTransform ?? DEFAULT_TRANSFORMS[0]
    return {
      v1: { x: transform.a, y: transform.c },
      v2: { x: transform.b, y: transform.d },
      translation: { x: transform.e, y: transform.f },
    }
  }, [selectedTransform])

  const handleMapChange = useCallback((value: LinearMapValue) => {
    setTransforms((prev) =>
      prev.map((transform, index) => {
        if (index !== selectedTransformIndex) return transform
        return {
          ...transform,
          a: value.v1.x,
          b: value.v2.x,
          c: value.v1.y,
          d: value.v2.y,
          e: value.translation.x,
          f: value.translation.y,
        }
      }),
    )
  }, [selectedTransformIndex])

  return (
    <div className="app-container">
      <header className="app-header">
        <div>
          <p className="eyebrow">IFS Playground</p>
          <h1>Barnsley Fern</h1>
          <p className="subtitle">Pyodide上のPythonでIFSを計算し、キャンバスに描画します。</p>
        </div>
      </header>

      <div className="app-body">
        <section className="canvas-panel">
          <div className="canvas-frame">
            <canvas ref={canvasRef} className="fern-canvas" />
          </div>
          <p className="status">{statusText}</p>
        </section>

        <aside className="controls-panel">
          <div className="control-card">
            <h2>Parameters</h2>
            <label className="control">
              <span>Points</span>
              <input
                type="range"
                min={10000}
                max={30000}
                step={1000}
                value={pointCount}
                onChange={(event) => setPointCount(Number(event.target.value))}
              />
              <span className="value">{pointCount.toLocaleString()}</span>
            </label>
            <label className="control">
              <span>Burn-in</span>
              <input
                type="range"
                min={0}
                max={500}
                step={10}
                value={burnIn}
                onChange={(event) => setBurnIn(Number(event.target.value))}
              />
              <span className="value">{burnIn.toLocaleString()}</span>
            </label>
            <label className="control">
              <span>Seed</span>
              <input
                type="number"
                min={0}
                max={1000000}
                value={seed}
                onChange={(event) => setSeed(Number(event.target.value))}
              />
            </label>
            <p className="hint">値を変えると自動で再計算します。</p>
          </div>
        </aside>
      </div>

      <section className="transform-editor">
        <div className="transform-list">
          <h2>IFS Maps</h2>
          <p className="hint">編集したい写像を選んで、右の線型写像UIで調整します。</p>
          <div className="transform-buttons">
            {transforms.map((transform, index) => (
              <button
                key={transform.id}
                className={`transform-button ${index === selectedTransformIndex ? 'is-active' : ''}`}
                onClick={() => setSelectedTransformIndex(index)}
                type="button"
              >
                <span>{transform.label}</span>
                <span className="probability">p={transform.p.toFixed(2)}</span>
              </button>
            ))}
          </div>
        </div>
        <div className="transform-visualizer">
          {selectedTransform && (
            <LinearMapVisualizer value={selectedMapValue} onChange={handleMapChange} showReset={false} />
          )}
        </div>
      </section>
    </div>
  )
}

export default App
