import { useCallback, useMemo, useRef, useState } from 'react'
import type { PreparedContour } from '../prepareContourForEscherization'
import { imageDataFromFile, prepareContourForEscherization } from '../prepareContourForEscherization'

export function ContourPrepTestPage() {
  const [file, setFile] = useState<File | null>(null)
  const [simplifyRatio, setSimplifyRatio] = useState(0.008)
  const [theta1Deg, setTheta1Deg] = useState(0)
  const [theta2Deg, setTheta2Deg] = useState(90)
  const [timeoutMs, setTimeoutMs] = useState(90_000)
  const [result, setResult] = useState<PreparedContour | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isRunning, setIsRunning] = useState(false)

  const canvasRef = useRef<HTMLCanvasElement>(null)

  const theta1 = useMemo(() => (theta1Deg * Math.PI) / 180, [theta1Deg])
  const theta2 = useMemo(() => (theta2Deg * Math.PI) / 180, [theta2Deg])

  const draw = useCallback((prepared: PreparedContour) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = prepared.imageData.width
    canvas.height = prepared.imageData.height
    ctx.putImageData(prepared.imageData, 0, 0)

    const contour = prepared.contour
    if (contour.length < 2) return

    ctx.save()
    ctx.lineWidth = Math.max(1, Math.floor(Math.min(canvas.width, canvas.height) / 400))
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.9)'
    ctx.beginPath()
    ctx.moveTo(contour[0][0], contour[0][1])
    for (let i = 1; i < contour.length; i++) ctx.lineTo(contour[i][0], contour[i][1])
    ctx.closePath()
    ctx.stroke()
    ctx.restore()
  }, [])

	  const run = useCallback(async () => {
	    if (!file) return
	    setIsRunning(true)
	    setError(null)
	    setResult(null)
	    try {
	      const imageData = await imageDataFromFile(file)
	      console.log('Loaded image:', imageData.width, 'x', imageData.height)
	      const prepared = await prepareContourForEscherization(imageData, {
	        simplifyRatio,
	        theta1,
	        theta2,
	        timeoutMs,
	      })
	      console.log('Prepared contour:', prepared)
	      setResult(prepared)
	      draw(prepared)
	    } catch (e) {
	      console.error('Contour prep failed:', e)
	      setError(e instanceof Error ? e.message : String(e))
	    } finally {
	      setIsRunning(false)
	    }
	  }, [draw, file, simplifyRatio, theta1, theta2, timeoutMs])

  return (
    <section>
      <h2>prepare_contour_for_escherization (Test)</h2>
      <p>画像から輪郭抽出→CCW統一→平行四辺形ベクトル算出（Web Worker + d3-contour使用）</p>

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
            <span>theta1 (deg)</span>
            <input type="number" step="1" value={theta1Deg} onChange={(e) => setTheta1Deg(Number(e.target.value))} />
          </label>
	          <label className="control">
	            <span>theta2 (deg)</span>
	            <input type="number" step="1" value={theta2Deg} onChange={(e) => setTheta2Deg(Number(e.target.value))} />
	          </label>
	          <label className="control">
	            <span>timeout (ms)</span>
	            <input
	              type="number"
	              step="1000"
	              min="1000"
	              value={timeoutMs}
	              onChange={(e) => setTimeoutMs(Number(e.target.value))}
	            />
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
          {!result ? (
            <p>画像を選んで「実行」を押してください。</p>
          ) : (
            <dl>
              <div className="kv">
                <dt>contour points</dt>
                <dd>{result.contour.length}</dd>
              </div>
              <div className="kv">
                <dt>image size</dt>
                <dd>
                  {result.imageData.width} × {result.imageData.height}
                </dd>
              </div>
              <div className="kv">
                <dt>tran_u</dt>
                <dd>
                  ({result.tranU[0].toFixed(2)}, {result.tranU[1].toFixed(2)})
                </dd>
              </div>
              <div className="kv">
                <dt>tran_v</dt>
                <dd>
                  ({result.tranV[0].toFixed(2)}, {result.tranV[1].toFixed(2)})
                </dd>
              </div>
            </dl>
          )}
        </div>
      </div>
    </section>
  )
}
