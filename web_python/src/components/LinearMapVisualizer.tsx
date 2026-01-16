import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './LinearMapVisualizer.css'

export type Vec2 = {
  x: number
  y: number
}

type DragTarget = 'v1' | 'v2' | 't' | null

type Layout = {
  width: number
  height: number
  centerX: number
  centerY: number
  scale: number
}

type PointerPosition = {
  x: number
  y: number
}

const GRID_COUNT = 4
const MAX_COORD = 3.6
const HANDLE_RADIUS = 14

export type LinearMapValue = {
  v1: Vec2
  v2: Vec2
  translation: Vec2
}

type LinearMapVisualizerProps = {
  value?: LinearMapValue
  onChange?: (value: LinearMapValue) => void
  showReset?: boolean
  resetLabel?: string
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function formatNumber(value: number) {
  return value.toFixed(2)
}

function LinearMapVisualizer({
  value,
  onChange,
  showReset = true,
  resetLabel = 'Reset',
}: LinearMapVisualizerProps) {
  const [v1, setV1] = useState<Vec2>(value?.v1 ?? { x: 1, y: 0 })
  const [v2, setV2] = useState<Vec2>(value?.v2 ?? { x: 0, y: 1 })
  const [translation, setTranslation] = useState<Vec2>(value?.translation ?? { x: 0, y: 0 })
  const [hoverTarget, setHoverTarget] = useState<DragTarget>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [zoom, setZoom] = useState(1)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const dragTargetRef = useRef<DragTarget>(null)

  const determinant = useMemo(() => v1.x * v2.y - v1.y * v2.x, [v1, v2])

  useEffect(() => {
    if (!value) return
    setV1(value.v1)
    setV2(value.v2)
    setTranslation(value.translation)
  }, [
    value?.v1.x,
    value?.v1.y,
    value?.v2.x,
    value?.v2.y,
    value?.translation.x,
    value?.translation.y,
  ])

  const getLayout = useCallback((canvas: HTMLCanvasElement): Layout => {
    const rect = canvas.getBoundingClientRect()
    const scale = (Math.min(rect.width, rect.height) / (GRID_COUNT * 2 + 2)) * zoom
    return {
      width: rect.width,
      height: rect.height,
      centerX: rect.width / 2,
      centerY: rect.height / 2,
      scale,
    }
  }, [zoom])

  const toScreen = useCallback((vector: Vec2, layout: Layout) => {
    return {
      x: layout.centerX + vector.x * layout.scale,
      y: layout.centerY - vector.y * layout.scale,
    }
  }, [])

  const toWorld = useCallback((x: number, y: number, layout: Layout): Vec2 => {
    return {
      x: (x - layout.centerX) / layout.scale,
      y: (layout.centerY - y) / layout.scale,
    }
  }, [])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.round(rect.width * dpr)
    canvas.height = Math.round(rect.height * dpr)

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    const layout = getLayout(canvas)

    ctx.clearRect(0, 0, layout.width, layout.height)
    ctx.fillStyle = '#0b0f14'
    ctx.fillRect(0, 0, layout.width, layout.height)

    const drawLine = (a: Vec2, b: Vec2, color: string, width = 1) => {
      const p0 = toScreen(a, layout)
      const p1 = toScreen(b, layout)
      ctx.strokeStyle = color
      ctx.lineWidth = width
      ctx.beginPath()
      ctx.moveTo(p0.x, p0.y)
      ctx.lineTo(p1.x, p1.y)
      ctx.stroke()
    }

    for (let i = -GRID_COUNT; i <= GRID_COUNT; i += 1) {
      const startV2 = {
        x: v1.x * i + v2.x * -GRID_COUNT,
        y: v1.y * i + v2.y * -GRID_COUNT,
      }
      const endV2 = {
        x: v1.x * i + v2.x * GRID_COUNT,
        y: v1.y * i + v2.y * GRID_COUNT,
      }
      drawLine(startV2, endV2, '#1b2a36')

      const startV1 = {
        x: v1.x * -GRID_COUNT + v2.x * i,
        y: v1.y * -GRID_COUNT + v2.y * i,
      }
      const endV1 = {
        x: v1.x * GRID_COUNT + v2.x * i,
        y: v1.y * GRID_COUNT + v2.y * i,
      }
      drawLine(startV1, endV1, '#1b2a36')
    }

    drawLine({ x: -GRID_COUNT - 1, y: 0 }, { x: GRID_COUNT + 1, y: 0 }, '#2f3944', 1.5)
    drawLine({ x: 0, y: -GRID_COUNT - 1 }, { x: 0, y: GRID_COUNT + 1 }, '#2f3944', 1.5)

    const origin = { x: 0, y: 0 }
    drawLine(origin, v1, '#ff6b6b', 2.5)
    drawLine(origin, v2, '#4cc9f0', 2.5)
    drawLine(origin, translation, '#7ee787', 2)

    const drawHandle = (vector: Vec2, color: string, label: string) => {
      const point = toScreen(vector, layout)
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.arc(point.x, point.y, 8, 0, Math.PI * 2)
      ctx.fill()

      ctx.strokeStyle = '#0b0f14'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(point.x, point.y, 8, 0, Math.PI * 2)
      ctx.stroke()

      ctx.fillStyle = '#e6edf3'
      ctx.font = '12px system-ui, sans-serif'
      ctx.fillText(label, point.x + 10, point.y - 10)
    }

    drawHandle(v1, '#ff6b6b', 'e1')
    drawHandle(v2, '#4cc9f0', 'e2')
    drawHandle(translation, '#7ee787', 't')
  }, [getLayout, toScreen, translation, v1, v2])

  useEffect(() => {
    draw()
  }, [draw])

  useEffect(() => {
    const handleResize = () => draw()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [draw])

  const emitChange = (nextV1: Vec2, nextV2: Vec2, nextTranslation: Vec2) => {
    if (onChange) {
      onChange({ v1: nextV1, v2: nextV2, translation: nextTranslation })
    }
  }

  const updateVector = (target: DragTarget, world: Vec2) => {
    const clamped = {
      x: clamp(world.x, -MAX_COORD, MAX_COORD),
      y: clamp(world.y, -MAX_COORD, MAX_COORD),
    }

    if (target === 'v1') {
      setV1(clamped)
      emitChange(clamped, v2, translation)
    }
    if (target === 'v2') {
      setV2(clamped)
      emitChange(v1, clamped, translation)
    }
    if (target === 't') {
      setTranslation(clamped)
      emitChange(v1, v2, clamped)
    }
  }

  const getPointerPosition = (event: React.PointerEvent<HTMLCanvasElement>): PointerPosition | null => {
    const canvas = canvasRef.current
    if (!canvas) return null
    const rect = canvas.getBoundingClientRect()
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    }
  }

  const getTargetAt = (pointer: PointerPosition, layout: Layout): DragTarget => {
    const v1Screen = toScreen(v1, layout)
    const v2Screen = toScreen(v2, layout)
    const tScreen = toScreen(translation, layout)

    const distV1 = Math.hypot(pointer.x - v1Screen.x, pointer.y - v1Screen.y)
    const distV2 = Math.hypot(pointer.x - v2Screen.x, pointer.y - v2Screen.y)
    const distT = Math.hypot(pointer.x - tScreen.x, pointer.y - tScreen.y)

    if (distV1 < HANDLE_RADIUS) return 'v1'
    if (distV2 < HANDLE_RADIUS) return 'v2'
    if (distT < HANDLE_RADIUS) return 't'
    return null
  }

  const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const pointer = getPointerPosition(event)
    if (!pointer) return

    const layout = getLayout(canvas)
    const target = getTargetAt(pointer, layout)

    if (!target) {
      dragTargetRef.current = null
      setIsDragging(false)
      return
    }

    dragTargetRef.current = target
    setHoverTarget(target)
    setIsDragging(true)

    canvas.setPointerCapture(event.pointerId)
    const world = toWorld(pointer.x, pointer.y, layout)
    updateVector(target, world)
  }

  const handlePointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const pointer = getPointerPosition(event)
    if (!pointer) return

    const layout = getLayout(canvas)

    if (dragTargetRef.current) {
      const world = toWorld(pointer.x, pointer.y, layout)
      updateVector(dragTargetRef.current, world)
      return
    }

    const target = getTargetAt(pointer, layout)
    setHoverTarget(target)
  }

  const handlePointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !dragTargetRef.current) return
    canvasRef.current.releasePointerCapture(event.pointerId)
    dragTargetRef.current = null
    setIsDragging(false)
  }

  const handlePointerLeave = () => {
    if (!dragTargetRef.current) {
      setHoverTarget(null)
    }
  }

  const handleWheel = (event: React.WheelEvent<HTMLCanvasElement>) => {
    event.preventDefault()
    const direction = event.deltaY > 0 ? -1 : 1
    const nextZoom = zoom * (direction > 0 ? 1.1 : 0.9)
    setZoom(Math.max(1, Math.min(3, nextZoom)))
  }

  const handleReset = () => {
    const nextV1 = { x: 1, y: 0 }
    const nextV2 = { x: 0, y: 1 }
    const nextTranslation = { x: 0, y: 0 }
    setV1(nextV1)
    setV2(nextV2)
    setTranslation(nextTranslation)
    setZoom(1)
    emitChange(nextV1, nextV2, nextTranslation)
  }

  const canvasClassName = [
    'map-canvas',
    hoverTarget ? 'is-hovering' : '',
    isDragging ? 'is-dragging' : '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <section className="linear-map">
      {showReset && (
        <div className="linear-map-toolbar">
          <button className="reset-button" onClick={handleReset}>{resetLabel}</button>
        </div>
      )}

      <div className="app-layout">
        <div className="canvas-panel">
          <div className="canvas-frame">
            <canvas
              ref={canvasRef}
              className={canvasClassName}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerLeave={handlePointerLeave}
              onPointerCancel={handlePointerUp}
              onWheel={handleWheel}
            />
          </div>
          <p className="hint">赤と青のハンドルをドラッグして写像を変更します。</p>
        </div>

        <aside className="info-panel">
          <div className="info-card">
            <h2>行列表記</h2>
            <div className="matrix">
              <div className="bracket">[</div>
              <div className="matrix-body matrix-body-affine">
                <div>{formatNumber(v1.x)}</div>
                <div>{formatNumber(v2.x)}</div>
                <div>{formatNumber(translation.x)}</div>
                <div>{formatNumber(v1.y)}</div>
                <div>{formatNumber(v2.y)}</div>
                <div>{formatNumber(translation.y)}</div>
              </div>
              <div className="bracket">]</div>
            </div>
            <p className="matrix-note">列ベクトル: e1, e2, t</p>
          </div>

          <div className="info-card">
            <h2>ベクトルの値</h2>
            <div className="vector-row">
              <span className="chip chip-red">e1</span>
              <span>({formatNumber(v1.x)}, {formatNumber(v1.y)})</span>
            </div>
            <div className="vector-row">
              <span className="chip chip-blue">e2</span>
              <span>({formatNumber(v2.x)}, {formatNumber(v2.y)})</span>
            </div>
            <div className="vector-row">
              <span className="chip chip-green">t</span>
              <span>({formatNumber(translation.x)}, {formatNumber(translation.y)})</span>
            </div>
          </div>

          <div className="info-card">
            <h2>行列式</h2>
            <p className="determinant">det = {formatNumber(determinant)}</p>
            <p className="matrix-note">正: 向きを保持 / 負: 反転 / 0: つぶれる</p>
          </div>
        </aside>
      </div>
    </section>
  )
}

export default LinearMapVisualizer
