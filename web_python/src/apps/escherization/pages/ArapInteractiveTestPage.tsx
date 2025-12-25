import { useCallback, useMemo, useRef, useState } from 'react'
import type { PreparedContour } from '../prepareContourForEscherization'
import { imageDataFromFile, prepareContourForEscherization } from '../prepareContourForEscherization'
import type { TriangulationResult } from '../triangulateBoundary'
import { triangulateBoundary } from '../triangulateBoundary'
import { arapMeshDeform } from '../arapMeshDeform'

type Vec2 = readonly [number, number]
type PickMode = 'fixed' | 'draggable'

function dist2(a: Vec2, b: Vec2): number {
  const dx = a[0] - b[0]
  const dy = a[1] - b[1]
  return dx * dx + dy * dy
}

function canvasPointFromMouseEvent(canvas: HTMLCanvasElement, e: React.MouseEvent): Vec2 {
  const rect = canvas.getBoundingClientRect()
  const sx = canvas.width / rect.width
  const sy = canvas.height / rect.height
  const x = (e.clientX - rect.left) * sx
  const y = (e.clientY - rect.top) * sy
  return [x, y]
}

function drawCircle(ctx: CanvasRenderingContext2D, p: Vec2, r: number, fill: string, stroke: string) {
  ctx.beginPath()
  ctx.arc(p[0], p[1], r, 0, Math.PI * 2)
  ctx.fillStyle = fill
  ctx.fill()
  ctx.lineWidth = 2
  ctx.strokeStyle = stroke
  ctx.stroke()
}

export function ArapInteractiveTestPage() {
  const [file, setFile] = useState<File | null>(null)
  const [simplifyRatio, setSimplifyRatio] = useState(0.008)
  const [triMaxArea, setTriMaxArea] = useState<string>('') // empty => auto
  const [triMaxPoints, setTriMaxPoints] = useState(4000)

  const [arapIters, setArapIters] = useState(5)
  const [cgIters, setCgIters] = useState(400)
  const [cgTol, setCgTol] = useState(1e-6)

  const [pickMode, setPickMode] = useState<PickMode>('draggable')

  const [prepared, setPrepared] = useState<PreparedContour | null>(null)
  const [mesh, setMesh] = useState<TriangulationResult | null>(null)
  const [boundaryUnique, setBoundaryUnique] = useState<number[]>([])

  const [fixedIndex, setFixedIndex] = useState<number | null>(null)
  const [draggableIndex, setDraggableIndex] = useState<number | null>(null)
  const [draggablePos, setDraggablePos] = useState<Vec2 | null>(null)
  const [deformedVertices, setDeformedVertices] = useState<Vec2[] | null>(null)

  const [error, setError] = useState<string | null>(null)
  const [isRunning, setIsRunning] = useState(false)

  const canvasRef = useRef<HTMLCanvasElement>(null)

  const draggingRef = useRef(false)
  const solvingRef = useRef(false)
  const pendingPosRef = useRef<Vec2 | null>(null)

  const parsedTriMaxArea = useMemo(() => {
    const v = Number(triMaxArea)
    return Number.isFinite(v) && v > 0 ? v : undefined
  }, [triMaxArea])

  const fixedPosition = useMemo<Vec2 | null>(() => {
    if (!mesh || fixedIndex === null) return null
    return mesh.vertices[fixedIndex]
  }, [fixedIndex, mesh])

  const draw = useCallback(
    (prep: PreparedContour, tri: TriangulationResult, verts: Vec2[], fixed: Vec2 | null, handle: Vec2 | null) => {
      const canvas = canvasRef.current
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      canvas.width = prep.imageData.width
      canvas.height = prep.imageData.height
      ctx.putImageData(prep.imageData, 0, 0)

      ctx.save()
      ctx.globalAlpha = 0.55
      ctx.strokeStyle = 'rgba(0, 102, 255, 1)'
      ctx.lineWidth = Math.max(1, Math.floor(Math.min(canvas.width, canvas.height) / 700))
      for (const [a, b, c] of tri.faces) {
        const p0 = verts[a]
        const p1 = verts[b]
        const p2 = verts[c]
        ctx.beginPath()
        ctx.moveTo(p0[0], p0[1])
        ctx.lineTo(p1[0], p1[1])
        ctx.lineTo(p2[0], p2[1])
        ctx.closePath()
        ctx.stroke()
      }
      ctx.restore()

      ctx.save()
      ctx.globalAlpha = 0.85
      ctx.strokeStyle = 'rgba(0,0,0,0.65)'
      ctx.lineWidth = Math.max(2, Math.floor(Math.min(canvas.width, canvas.height) / 450))
      const contour = prep.contour
      if (contour.length >= 2) {
        ctx.beginPath()
        ctx.moveTo(contour[0][0], contour[0][1])
        for (let i = 1; i < contour.length; i++) ctx.lineTo(contour[i][0], contour[i][1])
        ctx.closePath()
        ctx.stroke()
      }
      ctx.restore()

      const r = Math.max(4, Math.floor(Math.min(canvas.width, canvas.height) / 180))
      if (fixed) drawCircle(ctx, fixed, r, 'rgba(255, 220, 0, 0.9)', 'rgba(0,0,0,0.7)')
      if (handle) drawCircle(ctx, handle, r, 'rgba(255, 0, 160, 0.9)', 'rgba(0,0,0,0.7)')
    },
    [],
  )

  const pickNearestBoundaryVertex = useCallback(
    (p: Vec2): number | null => {
      if (!mesh || boundaryUnique.length === 0) return null
      let best: number | null = null
      let bestD2 = Infinity
      for (const idx of boundaryUnique) {
        const d = dist2(mesh.vertices[idx], p)
        if (d < bestD2) {
          bestD2 = d
          best = idx
        }
      }
      const canvas = canvasRef.current
      if (!canvas) return best
      const pxThreshold = Math.max(10, Math.min(canvas.width, canvas.height) / 40)
      return bestD2 <= pxThreshold * pxThreshold ? best : null
    },
    [boundaryUnique, mesh],
  )

  const solveOnce = useCallback(
    async (targetPos: Vec2) => {
      if (!mesh || !deformedVertices || fixedIndex === null || draggableIndex === null) return
      if (fixedIndex === draggableIndex) return
      if (!fixedPosition) return

      solvingRef.current = true
      try {
        const result = await arapMeshDeform(
          mesh.vertices,
          mesh.faces,
          [
            { index: fixedIndex, position: fixedPosition },
            { index: draggableIndex, position: targetPos },
          ],
          { iterations: arapIters, cgMaxIterations: cgIters, cgTolerance: cgTol },
        )
        setDeformedVertices(result.deformedVertices)
        if (prepared) draw(prepared, mesh, result.deformedVertices, fixedPosition, targetPos)
      } finally {
        solvingRef.current = false
      }
    },
    [arapIters, cgIters, cgTol, deformedVertices, draw, draggableIndex, fixedIndex, fixedPosition, mesh, prepared],
  )

  const scheduleSolve = useCallback(
    (targetPos: Vec2) => {
      pendingPosRef.current = targetPos
      if (solvingRef.current) return

      const loop = async () => {
        const p = pendingPosRef.current
        if (!p) return
        pendingPosRef.current = null
        await solveOnce(p)
        if (pendingPosRef.current) await loop()
      }

      void loop()
    },
    [solveOnce],
  )

  const run = useCallback(async () => {
    if (!file) return
    setIsRunning(true)
    setError(null)
    setPrepared(null)
    setMesh(null)
    setDeformedVertices(null)
    setFixedIndex(null)
    setDraggableIndex(null)
    setDraggablePos(null)
    setBoundaryUnique([])

    try {
      const imageData = await imageDataFromFile(file)
      const prep = await prepareContourForEscherization(imageData, { simplifyRatio })
      const tri = triangulateBoundary(prep.contour, { maxArea: parsedTriMaxArea, maxPoints: triMaxPoints })

      const uniq = Array.from(new Set(tri.boundaryIndices))
      if (uniq.length < 2) throw new Error('boundary vertices not found')

      const fixed = uniq[0]
      const handle = uniq[Math.floor(uniq.length / 2)]
      const handlePos = tri.vertices[handle]

      setPrepared(prep)
      setMesh(tri)
      setBoundaryUnique(uniq)
      setFixedIndex(fixed)
      setDraggableIndex(handle)
      setDraggablePos(handlePos)
      setDeformedVertices(tri.vertices)

      draw(prep, tri, tri.vertices, tri.vertices[fixed], handlePos)
    } catch (e) {
      console.error('ARAP interactive test failed:', e)
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setIsRunning(false)
    }
  }, [draw, file, parsedTriMaxArea, simplifyRatio, triMaxPoints])

  const onCanvasMouseDown = useCallback(
    (e: React.MouseEvent) => {
      const canvas = canvasRef.current
      if (!canvas || !mesh) return
      if (!deformedVertices) return

      const p = canvasPointFromMouseEvent(canvas, e)
      const picked = pickNearestBoundaryVertex(p)
      if (picked === null) return

      if (pickMode === 'fixed') {
        if (draggableIndex !== null && picked === draggableIndex) return
        setFixedIndex(picked)
        if (prepared && draggablePos) draw(prepared, mesh, deformedVertices, mesh.vertices[picked], draggablePos)
        return
      }

      // draggable
      if (fixedIndex !== null && picked === fixedIndex) return
      setDraggableIndex(picked)
      const handlePos = deformedVertices[picked]
      setDraggablePos(handlePos)
      if (prepared) draw(prepared, mesh, deformedVertices, fixedPosition, handlePos)

      // start drag if clicking close
      draggingRef.current = true
    },
    [deformedVertices, draggableIndex, draggablePos, draw, fixedIndex, fixedPosition, mesh, pickMode, pickNearestBoundaryVertex, prepared],
  )

  const onCanvasMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const canvas = canvasRef.current
      if (!canvas) return
      if (!mesh || !prepared) return
      if (!draggingRef.current) return
      if (fixedIndex === null || draggableIndex === null) return
      if (fixedIndex === draggableIndex) return

      const p = canvasPointFromMouseEvent(canvas, e)
      setDraggablePos(p)
      scheduleSolve(p)
    },
    [draggableIndex, fixedIndex, mesh, prepared, scheduleSolve],
  )

  const onCanvasMouseUp = useCallback(() => {
    draggingRef.current = false
  }, [])

  return (
    <section>
      <h2>ARAP (Interactive)</h2>
      <p>境界上の固定点1つ + ドラッグ点1つでメッシュをARAP変形（`arap-mesh-deform`実装を2D/Worker向けに移植）</p>

      <div className="escher-controls">
        <label className="control">
          <span>画像ファイル</span>
          <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </label>

        <div className="controls-row">
          <label className="control">
            <span>simplifyRatio</span>
            <input type="number" step="0.001" min="0" value={simplifyRatio} onChange={(e) => setSimplifyRatio(Number(e.target.value))} />
          </label>
          <label className="control">
            <span>tri maxArea (empty=auto)</span>
            <input type="number" step="1" min="0" value={triMaxArea} onChange={(e) => setTriMaxArea(e.target.value)} />
          </label>
          <label className="control">
            <span>tri maxPoints</span>
            <input type="number" step="100" min="0" value={triMaxPoints} onChange={(e) => setTriMaxPoints(Number(e.target.value))} />
          </label>
          <label className="control">
            <span>ARAP iters</span>
            <input type="number" step="1" min="1" value={arapIters} onChange={(e) => setArapIters(Number(e.target.value))} />
          </label>
          <label className="control">
            <span>CG iters</span>
            <input type="number" step="50" min="1" value={cgIters} onChange={(e) => setCgIters(Number(e.target.value))} />
          </label>
          <label className="control">
            <span>CG tol</span>
            <input type="number" step="1e-6" min="0" value={cgTol} onChange={(e) => setCgTol(Number(e.target.value))} />
          </label>

          <button className="run-button" onClick={run} disabled={!file || isRunning}>
            {isRunning ? '処理中...' : 'メッシュ生成'}
          </button>
        </div>

        <div className="controls-row">
          <label className="control">
            <span>ピックモード</span>
            <select value={pickMode} onChange={(e) => setPickMode(e.target.value as PickMode)}>
              <option value="draggable">Draggable（クリックで選択→そのままドラッグ）</option>
              <option value="fixed">Fixed（クリックで固定点を設定）</option>
            </select>
          </label>
          <div className="control">
            <span>fixed index</span>
            <input value={fixedIndex ?? ''} readOnly />
          </div>
          <div className="control">
            <span>draggable index</span>
            <input value={draggableIndex ?? ''} readOnly />
          </div>
        </div>

        {error && <div className="error">エラー: {error}</div>}
      </div>

      <div className="escher-output">
        <div className="canvas-wrap">
          <canvas ref={canvasRef} onMouseDown={onCanvasMouseDown} onMouseMove={onCanvasMouseMove} onMouseUp={onCanvasMouseUp} onMouseLeave={onCanvasMouseUp} />
        </div>

        <div className="stats">
          <h3>状態</h3>
          {!prepared || !mesh ? (
            <p>「メッシュ生成」を押してください。</p>
          ) : (
            <dl>
              <div className="kv">
                <dt>contour points</dt>
                <dd>{prepared.contour.length}</dd>
              </div>
              <div className="kv">
                <dt>vertices</dt>
                <dd>{mesh.vertices.length}</dd>
              </div>
              <div className="kv">
                <dt>triangles</dt>
                <dd>{mesh.faces.length}</dd>
              </div>
              <div className="kv">
                <dt>boundary unique</dt>
                <dd>{boundaryUnique.length}</dd>
              </div>
            </dl>
          )}
        </div>
      </div>
    </section>
  )
}

