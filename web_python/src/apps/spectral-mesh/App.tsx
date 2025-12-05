import { useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid } from '@react-three/drei'
import MeshViewer, { type MeshData } from '../../components/MeshViewer'
import ControlPanel from './ControlPanel'
import { usePyodide } from '../../hooks/usePyodide'
import './App.css'

function App() {
  const [originalMesh, setOriginalMesh] = useState<MeshData | null>(null)
  const [filteredMesh, setFilteredMesh] = useState<MeshData | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [cutoffRatio, setCutoffRatio] = useState(0.3)
  const [numEigenpairs, setNumEigenpairs] = useState(100)
  
  const { pyodide, isReady, error: pyodideError } = usePyodide([
    'spectral_mesh_processing.py'
  ])

  const handleFileSelect = async (file: File) => {
    try {
      const text = await file.text()
      const mesh = parseOBJ(text)
      setOriginalMesh(mesh)
      setFilteredMesh(null)
    } catch (error) {
      console.error('Failed to load model:', error)
    }
  }

  const handleFilter = async () => {
    if (!originalMesh || !pyodide || !isReady) {
      return
    }

    setIsProcessing(true)

    try {
      // Pythonコードを実行
      await pyodide.runPythonAsync(`
import numpy as np
from spectral_mesh_processing import SpectralMeshProcessor

# メッシュデータを設定
vertices = np.array(${JSON.stringify(originalMesh.vertices)})
faces = np.array(${JSON.stringify(originalMesh.faces)})

print(f"頂点数: {len(vertices)}, 面数: {len(faces)}")

# スペクトル処理
processor = SpectralMeshProcessor(vertices, faces, dual_type='circumcentric')

# スペクトル分解
print(f"スペクトル分解開始 (${numEigenpairs}固有対)...")
processor.compute_spectrum(num_eigenpairs=${numEigenpairs}, verbose=False)

# ローパスフィルター適用
cutoff_freq = processor.nyquist_frequency * ${cutoffRatio}
print(f"ローパスフィルター適用 (カットオフ: {cutoff_freq:.2f})...")
filtered_vertices = processor.apply_lowpass_filter(
    cutoff_freq=cutoff_freq,
    include_residual=False,
    verbose=False
)

# 結果を格納
result_vertices = filtered_vertices.tolist()
result_faces = faces.tolist()

print("処理完了")
      `)

      // 結果を取得
      const resultVertices = pyodide.globals.get('result_vertices').toJs()
      const resultFaces = pyodide.globals.get('result_faces').toJs()

      setFilteredMesh({
        vertices: Array.from(resultVertices),
        faces: Array.from(resultFaces)
      })
    } catch (error) {
      console.error('Filter error:', error)
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="app-container">
      <ControlPanel
        onFileSelect={handleFileSelect}
        onFilter={handleFilter}
        cutoffRatio={cutoffRatio}
        onCutoffChange={setCutoffRatio}
        numEigenpairs={numEigenpairs}
        onNumEigenpairsChange={setNumEigenpairs}
        isProcessing={isProcessing}
        isReady={isReady}
      />

      <div className="canvas-container">
        {pyodideError && (
          <div style={{ color: 'red', padding: '1rem' }}>
            Error: {pyodideError}
          </div>
        )}
        <Canvas camera={{ position: [2, 2, 2], fov: 50 }}>
          <color attach="background" args={['#0a0a0a']} />
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 10, 5]} intensity={1} />
          <directionalLight position={[-10, -10, -5]} intensity={0.3} />
          
          {originalMesh && (
            <MeshViewer
              mesh={originalMesh}
              color="#3b82f6"
              position={[-0.6, 0, 0]}
              label="Original"
            />
          )}
          
          {filteredMesh && (
            <MeshViewer
              mesh={filteredMesh}
              color="#22c55e"
              position={[0.6, 0, 0]}
              label="Filtered"
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

function parseOBJ(text: string): MeshData {
  const vertices: number[][] = []
  const faces: number[][] = []
  
  const lines = text.split('\n')
  
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue
    
    const parts = trimmed.split(/\s+/)
    
    if (parts[0] === 'v') {
      vertices.push([
        parseFloat(parts[1]),
        parseFloat(parts[2]),
        parseFloat(parts[3])
      ])
    } else if (parts[0] === 'f') {
      const faceIndices = parts.slice(1, 4).map(p => {
        const idx = parseInt(p.split('/')[0]) - 1 // OBJは1-indexed
        return idx
      })
      if (faceIndices.length === 3) {
        faces.push(faceIndices)
      }
    }
  }
  
  return { vertices, faces }
}

export default App
