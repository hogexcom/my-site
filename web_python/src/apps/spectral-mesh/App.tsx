import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls, Grid } from '@react-three/drei'
import MeshViewer, { type MeshData, type BoundingInfo, computeBoundingInfo } from '../../components/MeshViewer'
import EigenvectorViewer from './EigenvectorViewer'
import ControlPanel from './ControlPanel'
import { type ControlPoint, evaluateCurve, calculateCurveAverage } from './CurveEditor'
import { loadMeshFile, loadMeshFromURL } from '../../utils/meshLoaders'
import { usePyodide } from '../../hooks/usePyodide'
import './App.css'

interface SpectrumInfo {
  maxEigenpairs: number
  nyquistFrequency: number
}

// カメラを再構成モデルの位置にフォーカスするコンポーネント
function CameraController({ targetPosition, enabled, onFocused }: { targetPosition: [number, number, number], enabled: boolean, onFocused: () => void }) {
  const { camera } = useThree()
  
  useEffect(() => {
    if (enabled) {
      // カメラを再構成モデルの方向に向ける（モデルの中心Y=0.75を見る）
      camera.position.set(targetPosition[0] + 3, 2.5, 4)
      camera.lookAt(targetPosition[0], 0.75, 0)
      // フォーカス完了を通知してフラグをリセット
      onFocused()
    }
  }, [camera, targetPosition, enabled, onFocused])
  
  return null
}

const defaultFilterCurve: ControlPoint[] = [
  { x: 0, y: 1 },
  { x: 1, y: 1 }
]

function App() {
  const [originalMesh, setOriginalMesh] = useState<MeshData | null>(null)
  const [reconstructedMesh, setReconstructedMesh] = useState<MeshData | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [numEigenpairs, setNumEigenpairs] = useState(100)
  const [filterCurvePoints, setFilterCurvePoints] = useState<ControlPoint[]>(defaultFilterCurve)
  const [isLoadingDefault, setIsLoadingDefault] = useState(true)
  const [spectrumInfo, setSpectrumInfo] = useState<SpectrumInfo | null>(null)
  const [analysisStatus, setAnalysisStatus] = useState<string>('')
  const [shouldFocusReconstructed, setShouldFocusReconstructed] = useState(false)
  const [selectedEigenvectorIndex, setSelectedEigenvectorIndex] = useState(0)
  const [currentEigenvector, setCurrentEigenvector] = useState<number[] | null>(null)
  const reconstructionTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const eigenvectorTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  
  const { pyodide, isReady, error: pyodideError } = usePyodide([
    'spectral_mesh_processing.py'
  ])

  // オリジナルメッシュのバウンディング情報を計算（再構成後も同じスケールで表示）
  const boundingInfo: BoundingInfo | undefined = useMemo(() => {
    if (!originalMesh) return undefined
    return computeBoundingInfo(originalMesh)
  }, [originalMesh])

  // デフォルトでbunny.objを読み込む
  useEffect(() => {
    const loadDefaultModel = async () => {
      try {
        const mesh = await loadMeshFromURL(`${import.meta.env.BASE_URL}bunny.obj`)
        setOriginalMesh(mesh)
      } catch (error) {
        console.error('Failed to load default model:', error)
      } finally {
        setIsLoadingDefault(false)
      }
    }
    loadDefaultModel()
  }, [])

  const handleFileSelect = async (file: File) => {
    try {
      const mesh = await loadMeshFile(file)
      setOriginalMesh(mesh)
      setReconstructedMesh(null)
      setSpectrumInfo(null)
      setShouldFocusReconstructed(false)
      setFilterCurvePoints(defaultFilterCurve)
    } catch (error) {
      console.error('Failed to load model:', error)
      alert(`ファイルの読み込みに失敗しました: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  // スペクトルから再構成（カーブフィルター適用 + 残差にカーブ平均値を適用）
  const reconstructMesh = useCallback(async (eigenpairs: number, curvePoints: ControlPoint[]) => {
    if (!pyodide || !isReady || !spectrumInfo || !originalMesh) return
    
    setIsProcessing(true)
    
    try {
      // カーブから各固有対に対するフィルター値を計算
      const filterValues: number[] = []
      for (let i = 0; i < eigenpairs; i++) {
        const x = eigenpairs > 1 ? i / (eigenpairs - 1) : 0
        filterValues.push(evaluateCurve(curvePoints, x))
      }
      
      // カーブの平均値（残差に適用）
      const curveAverage = calculateCurveAverage(curvePoints)
      
      console.log('Filter values sample:', filterValues.slice(0, 5), '... length:', filterValues.length)
      console.log('Curve average:', curveAverage)
      
      await pyodide.runPythonAsync(`
import numpy as np

# 指定された固有対数とフィルター値で再構成
num_eigenpairs_to_use = ${eigenpairs}
filter_values = np.array(${JSON.stringify(filterValues)})
curve_average = ${curveAverage}

# フーリエ係数にフィルター値を適用
# fourier_X, fourier_Y, fourier_Z は compute_spectrum で計算済み
filtered_fourier_X = filter_values[:num_eigenpairs_to_use] * processor.fourier_X[:num_eigenpairs_to_use]
filtered_fourier_Y = filter_values[:num_eigenpairs_to_use] * processor.fourier_Y[:num_eigenpairs_to_use]
filtered_fourier_Z = filter_values[:num_eigenpairs_to_use] * processor.fourier_Z[:num_eigenpairs_to_use]

# 逆MHT（逆変換）で頂点座標を再構成
eigenvectors_subset = processor.eigenvectors[:, :num_eigenpairs_to_use]
X_reconstructed = eigenvectors_subset @ filtered_fourier_X
Y_reconstructed = eigenvectors_subset @ filtered_fourier_Y
Z_reconstructed = eigenvectors_subset @ filtered_fourier_Z

reconstructed = np.column_stack([X_reconstructed, Y_reconstructed, Z_reconstructed])

# 再構成 = フィルター適用済み成分 + 残差 * カーブ平均値
reconstructed_vertices = (reconstructed + fixed_residual * curve_average).tolist()
reconstructed_faces = faces.tolist()

print(f"再構成完了: 固有対数={num_eigenpairs_to_use}, フィルター平均={np.mean(filter_values):.4f}, 残差ゲイン={curve_average:.4f}")
      `)

      const resultVertices = pyodide.globals.get('reconstructed_vertices').toJs()
      const resultFaces = pyodide.globals.get('reconstructed_faces').toJs()

      setReconstructedMesh({
        vertices: Array.from(resultVertices),
        faces: Array.from(resultFaces)
      })
    } catch (error) {
      console.error('Reconstruction error:', error)
    } finally {
      setIsProcessing(false)
    }
  }, [pyodide, isReady, spectrumInfo, originalMesh])

  // スペクトル解析（固有値・固有ベクトル計算）
  const handleAnalyze = async () => {
    if (!originalMesh || !pyodide || !isReady) {
      return
    }

    setIsAnalyzing(true)
    setReconstructedMesh(null)

    try {
      // Step 1: Laplacian行列の構築
      setAnalysisStatus('Step 1/2: Laplacian行列を構築中...')
      // UIを更新させるために少し待機
      await new Promise(resolve => setTimeout(resolve, 50))
      
      await pyodide.runPythonAsync(`
import numpy as np
from spectral_mesh_processing import SpectralMeshProcessor

# メッシュデータを設定
vertices = np.array(${JSON.stringify(originalMesh.vertices)})
faces = np.array(${JSON.stringify(originalMesh.faces)})
print(f"頂点数: {len(vertices)}, 面数: {len(faces)}")

# スペクトル処理クラスを初期化
processor = SpectralMeshProcessor(vertices, faces, dual_type='circumcentric')
# Laplacian行列を構築
processor.compute_laplacian()
print("Laplacian行列の構築完了")
      `)
      
      // Step 2: 固有値・固有ベクトル・スペクトル係数の計算
      setAnalysisStatus('Step 2/2: 固有値・固有ベクトル・スペクトル係数を計算中... (数分かかる場合があります)')
      // UIを更新させるために少し待機
      await new Promise(resolve => setTimeout(resolve, 50))
      await pyodide.runPythonAsync(`
# スペクトル分解（固有対数は自動決定、スペクトル係数も計算）
processor.compute_spectrum(verbose=False)

# 結果を格納
spectrum_max_eigenpairs = len(processor.eigenvalues)
spectrum_nyquist_frequency = float(processor.nyquist_frequency)

# 固定の残差を計算（全固有対で再構成してもオリジナルと一致しない部分）
# これは固有対数に関係なく一定
max_freq_all = processor.frequencies[-1]
full_reconstruction = processor.apply_lowpass_filter(
    cutoff_freq=max_freq_all,
    include_residual=False,
    verbose=False
)
fixed_residual = vertices - full_reconstruction
print(f"残差のノルム: {np.linalg.norm(fixed_residual):.6f}")

print(f"解析完了: 固有対数={spectrum_max_eigenpairs}, ナイキスト周波数={spectrum_nyquist_frequency:.2f}")
      `)

      const maxEigenpairs = pyodide.globals.get('spectrum_max_eigenpairs')
      const nyquistFrequency = pyodide.globals.get('spectrum_nyquist_frequency')

      setSpectrumInfo({
        maxEigenpairs,
        nyquistFrequency
      })
      // スライダーを最大値に設定
      setNumEigenpairs(maxEigenpairs)
      setAnalysisStatus(`✅ 解析完了: ${maxEigenpairs}固有対, ナイキスト周波数=${nyquistFrequency.toFixed(2)}`)
      
      // 解析完了後にフィルターなしで再構成（全固有対 + 残差）
      setIsAnalyzing(false)
      setIsProcessing(true)
      
      try {
        await pyodide.runPythonAsync(`
# フィルターなしで再構成（全固有対を使用）
X_reconstructed = processor.eigenvectors @ processor.fourier_X
Y_reconstructed = processor.eigenvectors @ processor.fourier_Y
Z_reconstructed = processor.eigenvectors @ processor.fourier_Z
reconstructed = np.column_stack([X_reconstructed, Y_reconstructed, Z_reconstructed])

# 残差を加える（= オリジナルと同じ）
reconstructed_vertices = (reconstructed + fixed_residual).tolist()
reconstructed_faces = faces.tolist()
print("初期再構成完了（フィルターなし）")
        `)
        
        const resultVertices = pyodide.globals.get('reconstructed_vertices').toJs()
        const resultFaces = pyodide.globals.get('reconstructed_faces').toJs()

        setReconstructedMesh({
          vertices: Array.from(resultVertices),
          faces: Array.from(resultFaces)
        })
      } finally {
        setIsProcessing(false)
      }
      
      setShouldFocusReconstructed(true)
      
    } catch (error) {
      console.error('Analyze error:', error)
      setAnalysisStatus('❌ エラーが発生しました')
      setIsAnalyzing(false)
    }
  }

  // 固有対数スライダー変更時に再構成（デバウンス付き）
  const handleEigenpairsChange = useCallback((value: number) => {
    setNumEigenpairs(value)
    
    // デバウンス: 前回のタイマーをクリア
    if (reconstructionTimeoutRef.current) {
      clearTimeout(reconstructionTimeoutRef.current)
    }
    
    // 300ms後に再構成
    reconstructionTimeoutRef.current = setTimeout(() => {
      reconstructMesh(value, filterCurvePoints)
    }, 300)
  }, [reconstructMesh, filterCurvePoints])

  // フィルターカーブ変更時に再構成（デバウンス付き）
  const handleFilterCurveChange = useCallback((points: ControlPoint[]) => {
    setFilterCurvePoints(points)
    
    // デバウンス: 前回のタイマーをクリア
    if (reconstructionTimeoutRef.current) {
      clearTimeout(reconstructionTimeoutRef.current)
    }
    
    // 300ms後に再構成
    reconstructionTimeoutRef.current = setTimeout(() => {
      reconstructMesh(numEigenpairs, points)
    }, 300)
  }, [reconstructMesh, numEigenpairs])

  // 固有ベクトルを取得
  const fetchEigenvector = useCallback(async (index: number) => {
    if (!pyodide || !isReady || !spectrumInfo) return
    
    try {
      await pyodide.runPythonAsync(`
# 指定されたインデックスの固有ベクトルを取得
eigenvector_index = ${index}
eigenvector_data = processor.eigenvectors[:, eigenvector_index].tolist()
      `)
      
      const eigenvectorData = pyodide.globals.get('eigenvector_data').toJs()
      setCurrentEigenvector(Array.from(eigenvectorData))
    } catch (error) {
      console.error('Failed to fetch eigenvector:', error)
    }
  }, [pyodide, isReady, spectrumInfo])

  // 固有ベクトルインデックス変更時（デバウンス付き）
  const handleEigenvectorIndexChange = useCallback((value: number) => {
    setSelectedEigenvectorIndex(value)
    
    // デバウンス: 前回のタイマーをクリア
    if (eigenvectorTimeoutRef.current) {
      clearTimeout(eigenvectorTimeoutRef.current)
    }
    
    // 100ms後に取得
    eigenvectorTimeoutRef.current = setTimeout(() => {
      fetchEigenvector(value)
    }, 100)
  }, [fetchEigenvector])

  // 解析完了時に最初の固有ベクトルを取得（φ₁から開始）
  useEffect(() => {
    if (spectrumInfo && pyodide && isReady) {
      setSelectedEigenvectorIndex(1)
      fetchEigenvector(1)
    }
  }, [spectrumInfo, pyodide, isReady, fetchEigenvector])

  // クリーンアップ
  useEffect(() => {
    return () => {
      if (reconstructionTimeoutRef.current) {
        clearTimeout(reconstructionTimeoutRef.current)
      }
      if (eigenvectorTimeoutRef.current) {
        clearTimeout(eigenvectorTimeoutRef.current)
      }
    }
  }, [])

  return (
    <div className="app-container">
      <ControlPanel
        onFileSelect={handleFileSelect}
        onAnalyze={handleAnalyze}
        numEigenpairs={numEigenpairs}
        onNumEigenpairsChange={handleEigenpairsChange}
        filterCurvePoints={filterCurvePoints}
        onFilterCurveChange={handleFilterCurveChange}
        selectedEigenvectorIndex={selectedEigenvectorIndex}
        onEigenvectorIndexChange={handleEigenvectorIndexChange}
        isProcessing={isProcessing}
        isAnalyzing={isAnalyzing}
        isReady={isReady}
        isLoadingDefault={isLoadingDefault}
        spectrumInfo={spectrumInfo}
        hasMesh={!!originalMesh}
        analysisStatus={analysisStatus}
      />

      <div className="canvas-container">
        {pyodideError && (
          <div style={{ color: 'red', padding: '1rem' }}>
            Error: {pyodideError}
          </div>
        )}
        <Canvas camera={{ position: [3, 2.5, 4], fov: 50 }}>
          <color attach="background" args={['#0a0a0a']} />
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 10, 5]} intensity={1} />
          <directionalLight position={[-10, -10, -5]} intensity={0.3} />
          
          <CameraController 
            targetPosition={[0.9, 0.75, 0]} 
            enabled={shouldFocusReconstructed && !!reconstructedMesh}
            onFocused={() => setShouldFocusReconstructed(false)}
          />
          
          {originalMesh && (
            <MeshViewer
              mesh={originalMesh}
              color="#3b82f6"
              position={reconstructedMesh ? [0, 0, 0] : [0, 0, 0]}
              label="Original"
              boundingInfo={boundingInfo}
            />
          )}
          
          {reconstructedMesh && (
            <MeshViewer
              mesh={reconstructedMesh}
              color="#22c55e"
              position={[1.8, 0, 0]}
              label={`Reconstructed (${numEigenpairs})`}
              boundingInfo={boundingInfo}
            />
          )}
          
          {/* 基底関数（固有ベクトル）表示 */}
          {originalMesh && currentEigenvector && spectrumInfo && (
            <EigenvectorViewer
              mesh={originalMesh}
              eigenvector={currentEigenvector}
              position={[-1.8, 0, 0]}
              label={`φ${selectedEigenvectorIndex}`}
              boundingInfo={boundingInfo}
            />
          )}
          
          <Grid args={[10, 10]} cellColor="#333" sectionColor="#555" />
          <OrbitControls 
            enableDamping
            dampingFactor={0.05}
            rotateSpeed={0.5}
            zoomSpeed={0.3}
            panSpeed={0.5}
          />
        </Canvas>
      </div>
      <footer className="app-footer">
        <p>
          <a href={import.meta.env.BASE_URL}>← Back to Apps</a>
        </p>
      </footer>
    </div>
  )
}

export default App
