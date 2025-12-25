import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { usePyodide } from '../../../hooks/usePyodide'
import { imageDataFromImageBitmap } from '../prepareContourForEscherization'
import type { EscherizationPipelineOptions, TilingPattern } from '../escherizationPipeline'
import { runEscherizationCore } from '../escherizationPipeline'
import { buildTilingTiles, drawTiledMesh } from '../tilingRenderer'

type CanvasSize = { width: number; height: number }
type CanvasView = { zoom: number; panX: number; panY: number; centerX?: number; centerY?: number }

const BASE_URL = import.meta.env.BASE_URL
const SAMPLE_IMAGES = [
  { value: 'yumekawa_animal_usagi.png', label: 'ゆめかわ うさぎ' },
  { value: 'asobu_cat_shadow.png', label: '遊ぶ ねこ' },
  { value: 'eto_uma_banzai.png', label: '干支 うま' },
  { value: 'eto_ushi_banzai.png', label: '干支 うし' },
  { value: 'point_katen_woman.png', label: 'ポイント 女性' },
  { value: 'seiji_souridaijin_bg3.png', label: '政治 総理大臣' },
] as const

export function EscherizationMainPage() {
  const pythonFiles = useMemo(
    () => [
      'escherization/escherize_boundary.py',
      'escherization/p1.py',
      'escherization/p2.py',
      'escherization/p3.py',
      'escherization/tiling_condition.py',
    ],
    [],
  )

  const { pyodide, isReady, error: pyodideError } = usePyodide(pythonFiles)

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const canvasWrapRef = useRef<HTMLDivElement | null>(null)

  const [activeSource, setActiveSource] = useState<'sample' | 'file'>('sample')
  const [file, setFile] = useState<File | null>(null)
  const [selectedSample, setSelectedSample] = useState<string>(SAMPLE_IMAGES[0].value)
  const [tilingPattern, setTilingPattern] = useState<TilingPattern>('P3')
  const [m, setM] = useState(21)
  const [n, setN] = useState(21)
  const [nx, setNx] = useState(5)
  const [ny, setNy] = useState(5)

  const [contourSimplifyRatio, setContourSimplifyRatio] = useState(0.005)
  const [triMaxArea] = useState<string>('') // empty => auto

  const [status, setStatus] = useState('待機中')
  const [error, setError] = useState<string | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState<Awaited<ReturnType<typeof runEscherizationCore>> | null>(null)
  const [sourceImage, setSourceImage] = useState<ImageBitmap | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [canvasSize, setCanvasSize] = useState<CanvasSize>({ width: 900, height: 600 })
  const [view, setView] = useState<CanvasView>({ zoom: 1, panX: 0, panY: 0 })
  const dragState = useRef<{ isDragging: boolean; lastX: number; lastY: number }>({
    isDragging: false,
    lastX: 0,
    lastY: 0,
  })
  const lastRunKeyRef = useRef<string | null>(null)
  const viewRef = useRef<CanvasView>(view)

  useEffect(() => {
    if (tilingPattern !== 'P2') return
    if (m % 2 === 0) setM((prev) => prev + 1)
    if (n % 2 === 0) setN((prev) => prev + 1)
  }, [m, n, tilingPattern])

  useEffect(() => {
    const wrap = canvasWrapRef.current
    if (!wrap) return

    const updateSize = () => {
      const nextWidth = Math.max(320, wrap.clientWidth)
      const nextHeight = Math.max(420, wrap.clientHeight || 640)
      setCanvasSize({ width: nextWidth, height: nextHeight })
    }

    updateSize()

    const observer = new ResizeObserver(updateSize)
    observer.observe(wrap)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    return () => {
      sourceImage?.close()
    }
  }, [sourceImage])

  useEffect(() => {
    if (activeSource === 'sample') {
      setPreviewUrl(`${BASE_URL}escherization/sample_images/${selectedSample}`)
      return
    }

    if (!file) {
      setPreviewUrl(null)
      return
    }

    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [activeSource, file, selectedSample])

  const parsedTriMaxArea = useMemo(() => {
    const v = Number(triMaxArea)
    return Number.isFinite(v) && v > 0 ? v : undefined
  }, [triMaxArea])

  const normalizedMN = useMemo(() => {
    if (tilingPattern !== 'P2') return { m, n }
    return {
      m: m % 2 === 0 ? m + 1 : m,
      n: n % 2 === 0 ? n + 1 : n,
    }
  }, [m, n, tilingPattern])

  const runWithImage = useCallback(
    async (bitmap: ImageBitmap, imageData: ImageData, sourceKey: string) => {
      if (!pyodide || !isReady) return

      setIsRunning(true)
      setError(null)
      setResult(null)

      try {
        setSourceImage(bitmap)
        const opts: EscherizationPipelineOptions = {
          tilingPattern,
          m: normalizedMN.m,
          n: normalizedMN.n,
          useResample: true,
          contourSimplifyRatio,
          triangulationMaxArea: parsedTriMaxArea,
          triangulationMaxPoints: 4000,
          arapIterations: 8,
          arapCgMaxIterations: 300,
          arapCgTolerance: 1e-6,
        }

        setStatus('輪郭抽出→Pythonエッシャー化→メッシュ→ASAP...')
        const core = await runEscherizationCore(imageData, pyodide, opts)
        setResult(core)
        lastRunKeyRef.current = `${tilingPattern}:${normalizedMN.m}:${normalizedMN.n}:${contourSimplifyRatio}:${sourceKey}`
        setStatus(`完了: verts=${core.triangulation.vertices.length}, tris=${core.triangulation.faces.length}`)
      } catch (e) {
        console.error('main pipeline failed', e)
        setError(e instanceof Error ? e.message : String(e))
        setStatus('失敗')
      } finally {
        setIsRunning(false)
      }
    },
    [contourSimplifyRatio, isReady, normalizedMN.m, normalizedMN.n, parsedTriMaxArea, pyodide, tilingPattern],
  )

  const run = useCallback(async () => {
    if (!pyodide || !isReady) return

    setStatus('画像読み込み...')
    if (activeSource === 'file') {
      if (!file) return
      const bitmap = await createImageBitmap(file)
      const imageData = imageDataFromImageBitmap(bitmap)
      await runWithImage(bitmap, imageData, 'file')
      return
    }

    const res = await fetch(`${BASE_URL}escherization/sample_images/${selectedSample}`)
    if (!res.ok) {
      throw new Error(`サンプル画像の取得に失敗しました: ${res.status} ${res.statusText}`)
    }
    const blob = await res.blob()
    const bitmap = await createImageBitmap(blob)
    const imageData = imageDataFromImageBitmap(bitmap)
    await runWithImage(bitmap, imageData, `sample:${selectedSample}`)
  }, [activeSource, file, isReady, pyodide, runWithImage, selectedSample])

  useEffect(() => {
    if (!pyodide || !isReady) return
    if (isRunning) return
    if (activeSource === 'file' && !file) return
    const sourceKey = activeSource === 'file' ? 'file' : `sample:${selectedSample}`
    const nextKey = `${tilingPattern}:${normalizedMN.m}:${normalizedMN.n}:${contourSimplifyRatio}:${sourceKey}`
    if (nextKey === lastRunKeyRef.current) return
    run()
  }, [
    activeSource,
    contourSimplifyRatio,
    file,
    isReady,
    isRunning,
    normalizedMN.m,
    normalizedMN.n,
    pyodide,
    run,
    selectedSample,
    tilingPattern,
  ])
  useEffect(() => {
    viewRef.current = view
  }, [view])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = canvas.getBoundingClientRect()
      const cursorX = e.clientX - rect.left
      const cursorY = e.clientY - rect.top
      const prev = viewRef.current
      const zoomFactor = Math.exp(-e.deltaY * 0.0015)
      const nextZoom = Math.min(6, Math.max(0.2, prev.zoom * zoomFactor))
      const factor = nextZoom / prev.zoom
      const nextPanX = cursorX - (cursorX - prev.panX) * factor
      const nextPanY = cursorY - (cursorY - prev.panY) * factor
      setView({ zoom: nextZoom, panX: nextPanX, panY: nextPanY })
    }

    canvas.addEventListener('wheel', onWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', onWheel)
  }, [])

  useEffect(() => {
    if (!result || !sourceImage) return
    const canvas = canvasRef.current
    if (!canvas) return
    canvas.width = canvasSize.width
    canvas.height = canvasSize.height
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const tiles = buildTilingTiles({
      tilingPattern,
      m: normalizedMN.m,
      n: normalizedMN.n,
      nx,
      ny,
      deformedVertices: result.arap.deformedVertices,
      escherizedBoundary: result.python.escherizedBoundary,
      transU: result.python.transU,
      transV: result.python.transV,
    })

    drawTiledMesh({
      ctx,
      image: sourceImage,
      sourceVertices: result.triangulation.vertices,
      faces: result.triangulation.faces,
      tiles,
      width: canvasSize.width,
      height: canvasSize.height,
      view,
    })
  }, [canvasSize.height, canvasSize.width, nx, ny, normalizedMN.m, normalizedMN.n, result, sourceImage, tilingPattern, view])

  const tilingDescription = (
    <p className="escher-help">
      P1は平行移動のみ、P2は180度回転＋平行移動（辺U/辺Vの頂点数は奇数に自動補正）、P3は120度回転＋平行移動（1辺の頂点数のみ使用）です。
    </p>
  )

  const tileMultiplier = tilingPattern === 'P2' ? 4 : tilingPattern === 'P3' ? 3 : 1
  const tileCountX = tilingPattern === 'P2' ? Math.max(1, Math.floor(nx / 2)) : Math.max(1, Math.floor(nx))
  const tileCountY = tilingPattern === 'P2' ? Math.max(1, Math.floor(ny / 2)) : Math.max(1, Math.floor(ny))

  return (
    <section className="escher-main-page">
      <header className="escher-main-hero">
        <div>
          <h2>Escherization Tiling（エッシャー化）</h2>
          <p>画像→輪郭→エッシャー化→メッシュ変形をWebで実行し、タイリング表示します。</p>
        </div>
      </header>

      <div className="escher-main-grid">
        <div className="escher-panel-stack">
          <div className="escher-panel">
            <h3>サンプル画像</h3>
            <label className="control">
              <span>サンプル選択</span>
              <select
                value={selectedSample}
                onChange={(e) => {
                  setSelectedSample(e.target.value)
                  setActiveSource('sample')
                }}
              >
                {SAMPLE_IMAGES.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="escher-panel">
            <h3>入力画像</h3>
            <label className="control">
              <span>画像ファイル</span>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => {
                  const nextFile = e.target.files?.[0] ?? null
                  setFile(nextFile)
                  setActiveSource('file')
                }}
              />
            </label>
          </div>

          <div className="escher-panel">
            <h3>タイリング設定</h3>
            <label className="control">
              <span>タイリングパターン</span>
              <select value={tilingPattern} onChange={(e) => setTilingPattern(e.target.value as TilingPattern)}>
                <option value="P2">P2</option>
                <option value="P3">P3</option>
                <option value="P1">P1</option>
              </select>
            </label>
            {tilingDescription}

            <div className="controls-row">
              {tilingPattern === 'P3' ? (
                <label className="control">
                  <span>1辺の頂点数</span>
                  <input type="number" step="1" min="3" value={m} onChange={(e) => setM(Number(e.target.value))} />
                </label>
              ) : (
                <>
                  <label className="control">
                    <span>辺Uの頂点数</span>
                    <input type="number" step="1" min="3" value={m} onChange={(e) => setM(Number(e.target.value))} />
                  </label>
                  <label className="control">
                    <span>辺Vの頂点数</span>
                    <input type="number" step="1" min="3" value={n} onChange={(e) => setN(Number(e.target.value))} />
                  </label>
                </>
              )}
            </div>

            <div className="controls-row">
              <label className="control">
                <span>タイル数 nx</span>
                <input type="number" step="1" min="1" value={nx} onChange={(e) => setNx(Number(e.target.value))} />
              </label>
              <label className="control">
                <span>タイル数 ny</span>
                <input type="number" step="1" min="1" value={ny} onChange={(e) => setNy(Number(e.target.value))} />
              </label>
            </div>
          </div>

          <div className="escher-panel">
            <h3>輪郭 &amp; メッシュ</h3>
            <div className="controls-row">
              <label className="control">
                <span>輪郭の細かさ</span>
                <select value={String(contourSimplifyRatio)} onChange={(e) => setContourSimplifyRatio(Number(e.target.value))}>
                  <option value="0.001">細かい (0.001)</option>
                  <option value="0.005">中 (0.005)</option>
                  <option value="0.01">荒い (0.01)</option>
                </select>
              </label>
            </div>
          </div>

          {pyodideError && <div className="error">Pyodideエラー: {pyodideError}</div>}
          {!pyodideError && !isReady && <div className="error">Pyodideロード中...</div>}
          {error && <div className="error">エラー: {error}</div>}

          <div className="escher-panel">
            <h3>実行ステータス</h3>
            <div className="control">
              <span>status</span>
              <input value={status} readOnly />
            </div>
            {tilingPattern === 'P2' && (
              <p className="escher-help">P2は奇数が必要なため、偶数は+1して自動補正します。</p>
            )}
          </div>
        </div>

        <div className="escher-tiling-panel">
          {previewUrl && (
            <div className="tiling-hud">
              <img src={previewUrl} alt="エッシャー化前の画像" />
            </div>
          )}
          <div className="escher-tiling-header">
            <div>
              <h3>タイリング表示</h3>
              <p>右側キャンバスにエッシャー化結果を敷き詰めます。</p>
            </div>
            {result && (
              <div className="tiling-stats">
                <span>verts: {result.triangulation.vertices.length}</span>
                <span>tris: {result.triangulation.faces.length}</span>
                <span>tiles: {tileCountX * tileCountY * tileMultiplier}</span>
              </div>
            )}
          </div>

          <div ref={canvasWrapRef} className="tiling-canvas-wrap">
            <canvas
              ref={canvasRef}
              className="tiling-canvas"
              onPointerDown={(e) => {
                dragState.current = { isDragging: true, lastX: e.clientX, lastY: e.clientY }
                ;(e.currentTarget as HTMLCanvasElement).setPointerCapture(e.pointerId)
              }}
              onPointerMove={(e) => {
                if (!dragState.current.isDragging) return
                const dx = e.clientX - dragState.current.lastX
                const dy = e.clientY - dragState.current.lastY
                dragState.current.lastX = e.clientX
                dragState.current.lastY = e.clientY
                setView((prev) => ({ ...prev, panX: prev.panX + dx, panY: prev.panY + dy }))
              }}
              onPointerUp={(e) => {
                dragState.current.isDragging = false
                ;(e.currentTarget as HTMLCanvasElement).releasePointerCapture(e.pointerId)
              }}
              onPointerLeave={() => {
                dragState.current.isDragging = false
              }}
            />
            {!result && <div className="tiling-empty">画像を選択して「エッシャー化を実行」を押してください。</div>}
          </div>
        </div>
      </div>
    </section>
  )
}
