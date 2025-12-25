import { useCallback, useMemo, useRef, useState } from 'react'
import type { PreparedContour } from '../prepareContourForEscherization'
import { imageDataFromFile, prepareContourForEscherization } from '../prepareContourForEscherization'
import type { TriangulationResult } from '../triangulateBoundary'
import { triangulateBoundary } from '../triangulateBoundary'

function triangleArea2(a: readonly [number, number], b: readonly [number, number], c: readonly [number, number]): number {
  return Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
}

export function TriangulationTestPage() {
  const [file, setFile] = useState<File | null>(null)
  const [simplifyRatio, setSimplifyRatio] = useState(0.008)
  const [maxArea, setMaxArea] = useState<string>('') // empty => auto
  const [maxPoints, setMaxPoints] = useState(10_000)
  const [timeoutMs, setTimeoutMs] = useState(90_000)

  const [prepared, setPrepared] = useState<PreparedContour | null>(null)
  const [mesh, setMesh] = useState<TriangulationResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isRunning, setIsRunning] = useState(false)

  const canvasRef = useRef<HTMLCanvasElement>(null)

  const parsedMaxArea = useMemo(() => {
    const v = Number(maxArea)
    return Number.isFinite(v) && v > 0 ? v : undefined
  }, [maxArea])

  const draw = useCallback((prep: PreparedContour, tri: TriangulationResult) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = prep.imageData.width
    canvas.height = prep.imageData.height
    ctx.putImageData(prep.imageData, 0, 0)

    ctx.save()
    ctx.lineWidth = Math.max(1, Math.floor(Math.min(canvas.width, canvas.height) / 700))
    ctx.globalAlpha = 0.45
    ctx.strokeStyle = 'rgba(0, 102, 255, 1)'

    for (const [a, b, c] of tri.faces) {
      const p0 = tri.vertices[a]
      const p1 = tri.vertices[b]
      const p2 = tri.vertices[c]
      ctx.beginPath()
      ctx.moveTo(p0[0], p0[1])
      ctx.lineTo(p1[0], p1[1])
      ctx.lineTo(p2[0], p2[1])
      ctx.closePath()
      ctx.stroke()
    }

    ctx.globalAlpha = 0.9
    ctx.strokeStyle = 'rgba(255, 0, 0, 1)'
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
  }, [])

  const run = useCallback(async () => {
    if (!file) return
    setIsRunning(true)
    setError(null)
    setPrepared(null)
    setMesh(null)

    try {
      const imageData = await imageDataFromFile(file)
      const prep = await prepareContourForEscherization(imageData, {
        simplifyRatio,
        timeoutMs,
      })
      const tri = triangulateBoundary(prep.contour, { maxArea: parsedMaxArea, maxPoints })

      setPrepared(prep)
      setMesh(tri)
      draw(prep, tri)
    } catch (e) {
      console.error('Triangulation test failed:', e)
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setIsRunning(false)
    }
  }, [draw, file, maxPoints, parsedMaxArea, simplifyRatio, timeoutMs])

  const stats = useMemo(() => {
    if (!mesh) return null
    let minArea = Infinity
    let maxArea = -Infinity
    let sumArea = 0
    for (const [a, b, c] of mesh.faces) {
      const area2 = triangleArea2(mesh.vertices[a], mesh.vertices[b], mesh.vertices[c])
      const area = area2 / 2
      if (area < minArea) minArea = area
      if (area > maxArea) maxArea = area
      sumArea += area
    }
    const meanArea = mesh.faces.length ? sumArea / mesh.faces.length : 0
    return {
      vertices: mesh.vertices.length,
      faces: mesh.faces.length,
      boundary: mesh.boundaryIndices.length,
      interior: mesh.vertices.length - mesh.boundaryIndices.length,
      minArea: Number.isFinite(minArea) ? minArea : 0,
      maxArea: Number.isFinite(maxArea) ? maxArea : 0,
      meanArea,
    }
  }, [mesh])

  return (
    <section>
      <h2>triangulate_boundary (Test)</h2>
      <p>輪郭点から三角形分割（poly2tri + Steiner点追加で面積制御を近似）</p>

      <div className="escher-controls">
        <label className="control">
          <span>画像ファイル</span>
          <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </label>

        <div className="controls-row">
          <label className="control">
            <span>simplifyRatio</span>
            <input
              type="number"
              step="0.001"
              min="0"
              value={simplifyRatio}
              onChange={(e) => setSimplifyRatio(Number(e.target.value))}
            />
          </label>
          <label className="control">
            <span>maxArea (empty=auto)</span>
            <input type="number" step="1" min="0" value={maxArea} onChange={(e) => setMaxArea(e.target.value)} />
          </label>
          <label className="control">
            <span>maxPoints</span>
            <input type="number" step="100" min="0" value={maxPoints} onChange={(e) => setMaxPoints(Number(e.target.value))} />
          </label>
          <label className="control">
            <span>timeout (ms)</span>
            <input type="number" step="1000" min="1000" value={timeoutMs} onChange={(e) => setTimeoutMs(Number(e.target.value))} />
          </label>

          <button className="run-button" onClick={run} disabled={!file || isRunning}>
            {isRunning ? '処理中...' : '実行'}
          </button>
        </div>

        {error && <div className="error">エラー: {error}</div>}
      </div>

      <div className="escher-output">
        <div className="canvas-wrap">
          <canvas ref={canvasRef} />
        </div>

        <div className="stats">
          <h3>結果</h3>
          {!prepared || !mesh || !stats ? (
            <p>画像を選んで「実行」を押してください。</p>
          ) : (
            <dl>
              <div className="kv">
                <dt>contour points</dt>
                <dd>{prepared.contour.length}</dd>
              </div>
              <div className="kv">
                <dt>vertices</dt>
                <dd>{stats.vertices}</dd>
              </div>
              <div className="kv">
                <dt>triangles</dt>
                <dd>{stats.faces}</dd>
              </div>
              <div className="kv">
                <dt>boundary</dt>
                <dd>
                  {stats.boundary} / interior {stats.interior}
                </dd>
              </div>
              <div className="kv">
                <dt>area (min/mean/max)</dt>
                <dd>
                  {stats.minArea.toFixed(2)} / {stats.meanArea.toFixed(2)} / {stats.maxArea.toFixed(2)}
                </dd>
              </div>
            </dl>
          )}
        </div>
      </div>
    </section>
  )
}
