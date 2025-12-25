import { useCallback, useMemo, useState } from 'react'
import { usePyodide } from '../../../hooks/usePyodide'
import { imageDataFromFile } from '../prepareContourForEscherization'
import type { EscherizationPipelineOptions, TilingPattern } from '../escherizationPipeline'
import { runEscherizationPipeline } from '../escherizationPipeline'

export function EscherizationPipelinePage() {
  const pythonFiles = useMemo(
    () => [
      'escherization/escherize_boundary.py',
      'escherization/p1.py',
      'escherization/p2.py',
      'escherization/p3.py',
      'escherization/tiling_condition.py',
      'escherization/visualize_matplotlib.py',
    ],
    [],
  )

  const { pyodide, isReady, error: pyodideError } = usePyodide(pythonFiles, { packages: ['matplotlib'] })

  const [file, setFile] = useState<File | null>(null)
  const [tilingPattern, setTilingPattern] = useState<TilingPattern>('P2')
  const [m, setM] = useState(21)
  const [n, setN] = useState(21)
  const [useResample, setUseResample] = useState(true)

  const [contourSimplifyRatio, setContourSimplifyRatio] = useState(0.008)
  const [triMaxArea] = useState<string>('') // empty => auto
  const [triMaxPoints, setTriMaxPoints] = useState(4000)

  const [arapIters, setArapIters] = useState(5)
  const [cgIters, setCgIters] = useState(400)
  const [cgTol, setCgTol] = useState(1e-6)

  const [pngBase64, setPngBase64] = useState<string | null>(null)
  const [status, setStatus] = useState<string>('待機中')
  const [error, setError] = useState<string | null>(null)
  const [isRunning, setIsRunning] = useState(false)

  const parsedTriMaxArea = useMemo(() => {
    const v = Number(triMaxArea)
    return Number.isFinite(v) && v > 0 ? v : undefined
  }, [triMaxArea])

  const run = useCallback(async () => {
    if (!file) return
    if (!pyodide || !isReady) return
    setIsRunning(true)
    setError(null)
    setPngBase64(null)

    try {
      setStatus('画像読み込み...')
      const imageData = await imageDataFromFile(file)

      const opts: EscherizationPipelineOptions = {
        tilingPattern,
        m,
        n,
        useResample,
        contourSimplifyRatio,
        triangulationMaxArea: parsedTriMaxArea,
        triangulationMaxPoints: triMaxPoints,
        arapIterations: arapIters,
        arapCgMaxIterations: cgIters,
        arapCgTolerance: cgTol,
      }

      setStatus('輪郭抽出→Pythonエッシャー化→メッシュ→ASAP→Matplotlib...')
      const result = await runEscherizationPipeline(imageData, pyodide, opts)
      setPngBase64(result.matplotlibPngBase64)
      setStatus(
        `完了: contour=${result.contour.contour.length}, xBest=${result.python.xBest.length}, verts=${result.triangulation.vertices.length}, tris=${result.triangulation.faces.length}`,
      )
    } catch (e) {
      console.error('pipeline failed', e)
      setError(e instanceof Error ? e.message : String(e))
      setStatus('失敗')
    } finally {
      setIsRunning(false)
    }
  }, [arapIters, cgIters, cgTol, contourSimplifyRatio, file, isReady, m, n, parsedTriMaxArea, pyodide, tilingPattern, triMaxPoints, useResample])

  return (
    <section>
      <h2>Matplotlib Debug Pipeline</h2>
      <p>Web実装（輪郭/メッシュ/ASAP）+ Python（escherize_boundary）で一連処理を実行し、Matplotlib結果を確認します。</p>

      <div className="escher-controls">
        <label className="control">
          <span>画像ファイル</span>
          <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </label>

        <div className="controls-row">
          <label className="control">
            <span>tilingPattern</span>
            <select value={tilingPattern} onChange={(e) => setTilingPattern(e.target.value as TilingPattern)}>
              <option value="P2">P2</option>
              <option value="P3">P3</option>
              <option value="P1">P1</option>
            </select>
          </label>
          <label className="control">
            <span>m</span>
            <input type="number" step="1" min="3" value={m} onChange={(e) => setM(Number(e.target.value))} />
          </label>
          <label className="control">
            <span>n</span>
            <input type="number" step="1" min="3" value={n} onChange={(e) => setN(Number(e.target.value))} />
          </label>
          <label className="control">
            <span>resample</span>
            <select value={useResample ? '1' : '0'} onChange={(e) => setUseResample(e.target.value === '1')}>
              <option value="1">true</option>
              <option value="0">false</option>
            </select>
          </label>
        </div>

        <div className="controls-row">
          <label className="control">
            <span>contour simplifyRatio</span>
            <input type="number" step="0.001" min="0" value={contourSimplifyRatio} onChange={(e) => setContourSimplifyRatio(Number(e.target.value))} />
          </label>
          <label className="control">
            <span>tri maxPoints</span>
            <input type="number" step="100" min="0" value={triMaxPoints} onChange={(e) => setTriMaxPoints(Number(e.target.value))} />
          </label>
        </div>

        <div className="controls-row">
          <label className="control">
            <span>ASAP iters</span>
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
        </div>

        {pyodideError && <div className="error">Pyodideエラー: {pyodideError}</div>}
        {!pyodideError && !isReady && <div className="error">Pyodideロード中...</div>}
        {error && <div className="error">エラー: {error}</div>}

        <div className="controls-row">
          <button className="run-button" onClick={run} disabled={!file || !isReady || !pyodide || isRunning}>
            {isRunning ? '処理中...' : '実行'}
          </button>
          <div className="control">
            <span>status</span>
            <input value={status} readOnly />
          </div>
        </div>
      </div>

      {pngBase64 && (
        <div className="canvas-wrap" style={{ marginTop: 16 }}>
          <img alt="matplotlib result" src={`data:image/png;base64,${pngBase64}`} style={{ maxWidth: '100%', height: 'auto', display: 'block' }} />
        </div>
      )}
    </section>
  )
}
