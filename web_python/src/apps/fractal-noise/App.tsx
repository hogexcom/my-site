import { useState, useCallback, useRef, useEffect } from 'react'
import { usePyodide } from '../../hooks/usePyodide'
import './App.css'

type TabType = '1d' | '2d' | 'raymarch'

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('1d')
  const [hurstIndex, setHurstIndex] = useState(0.75)
  const [resolution1D, setResolution1D] = useState(1024)
  const [resolution2D, setResolution2D] = useState(128)
  const [isGenerating, setIsGenerating] = useState(false)
  
  const canvas1DRef = useRef<HTMLCanvasElement>(null)
  const canvas2DRef = useRef<HTMLCanvasElement>(null)
  const canvasRaymarchRef = useRef<HTMLCanvasElement>(null)
  const heightmapRef = useRef<ImageData | null>(null)
  const animationFrameRef = useRef<number>(null)
  
  const { pyodide, isReady, error: pyodideError } = usePyodide([
    'fractal_noise_1d.py',
    'fractal_noise_2d.py'
  ])

  // 1D Fractal Noise Generation
  const generate1D = useCallback(async () => {
    if (!pyodide || !isReady || !canvas1DRef.current) return
    
    setIsGenerating(true)
    
    try {
      await pyodide.runPythonAsync(`
import fractal_noise_1d
import numpy as np

f = fractal_noise_1d.generate_fractal_1d(${resolution1D}, ${hurstIndex})
signal_data = f.tolist()
      `)
      
      const signalData = pyodide.globals.get('signal_data').toJs()
      const signal = Array.from(signalData) as number[]
      
      // Draw to canvas
      const canvas = canvas1DRef.current
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      
      canvas.width = 800
      canvas.height = 400
      
      ctx.fillStyle = '#1a1a1a'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      
      // Find min/max for scaling
      const min = Math.min(...signal)
      const max = Math.max(...signal)
      const range = max - min
      
      // Draw signal
      ctx.strokeStyle = '#00d4ff'
      ctx.lineWidth = 2
      ctx.beginPath()
      
      signal.forEach((value, i) => {
        const x = (i / signal.length) * canvas.width
        const y = canvas.height - ((value - min) / range) * canvas.height * 0.8 - canvas.height * 0.1
        
        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })
      
      ctx.stroke()
      
      // Add grid
      ctx.strokeStyle = '#333'
      ctx.lineWidth = 1
      for (let i = 0; i <= 4; i++) {
        const y = (i / 4) * canvas.height
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(canvas.width, y)
        ctx.stroke()
      }
      
    } catch (error) {
      console.error('1D Generation error:', error)
    } finally {
      setIsGenerating(false)
    }
  }, [pyodide, isReady, hurstIndex, resolution1D])

  // 2D Fractal Noise Generation
  const generate2D = useCallback(async (useHighRes = false) => {
    if (!pyodide || !isReady) return false
    
    setIsGenerating(true)
    
    // Raymarchタブの場合は1024x1024を使用
    const res = useHighRes ? 1024 : resolution2D
    
    try {
      await pyodide.runPythonAsync(`
import fractal_noise_2d
import numpy as np

heightmap = fractal_noise_2d.generate_fractal_2d(${res}, ${hurstIndex})
heightmap_normalized = ((heightmap - heightmap.min()) / (heightmap.max() - heightmap.min()) * 255).astype(np.uint8)
heightmap_list = heightmap_normalized.flatten().tolist()
heightmap_3d = heightmap.tolist()
      `)
      
      const heightmapList = pyodide.globals.get('heightmap_list').toJs()
      const heightmap3D = pyodide.globals.get('heightmap_3d').toJs()
      const heightmapArray = new Uint8ClampedArray(Array.from(heightmapList) as number[])
      
      // Create ImageData for heightmap (store for raymarch)
      const imageData = new ImageData(res, res)
      for (let i = 0; i < heightmapArray.length; i++) {
        imageData.data[i * 4] = heightmapArray[i]
        imageData.data[i * 4 + 1] = heightmapArray[i]
        imageData.data[i * 4 + 2] = heightmapArray[i]
        imageData.data[i * 4 + 3] = 255
      }
      heightmapRef.current = imageData
      
      // Draw 3D mesh using simple projection (only if canvas is available)
      const canvas = canvas2DRef.current
      if (!canvas) {
        // No canvas for 2D view, but heightmap is ready
        return true
      }
      
      const ctx = canvas.getContext('2d')
      if (!ctx) return true
      
      canvas.width = 800
      canvas.height = 600
      
      ctx.fillStyle = '#1a1a1a'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      
      const heightmap3DArray = Array.from(heightmap3D).map(row => Array.from(row as any) as number[])
      
      // Find min/max for proper scaling
      let minH = Infinity
      let maxH = -Infinity
      for (let y = 0; y < res; y++) {
        for (let x = 0; x < res; x++) {
          const h = heightmap3DArray[y][x]
          if (h < minH) minH = h
          if (h > maxH) maxH = h
        }
      }
      const heightRange = maxH - minH
      
      // Simple isometric projection
      const scale = Math.min(canvas.width, canvas.height) / res / 1.5
      const heightScale = heightRange > 0 ? 200 / heightRange : 100
      const yOffset = -100 // メッシュ全体を上に移動
      
      console.log(`Height range: ${minH.toFixed(3)} to ${maxH.toFixed(3)}, scale: ${heightScale.toFixed(2)}`)
      
      // Draw wireframe (ステップサイズを固定して解像度に応じて線が増える)
      const step = 8 // 固定ステップ：64pxなら8本、64pxなら64本
      ctx.strokeStyle = '#00d4ff'
      ctx.lineWidth = 0.5
      
      // Rows (x方向に線を引く)
      for (let y = 0; y < res; y += step) {
        ctx.beginPath()
        for (let x = 0; x < res; x += step) {
          const h = (heightmap3DArray[y][x] - minH) * heightScale
          const px = canvas.width / 2 + (x - y) * scale * 0.866
          const py = canvas.height / 2 + (x + y) * scale * 0.5 - h + yOffset
          
          if (x === 0) {
            ctx.moveTo(px, py)
          } else {
            ctx.lineTo(px, py)
          }
        }
        ctx.stroke()
      }
      
      // Columns (y方向に線を引く)
      for (let x = 0; x < res; x += step) {
        ctx.beginPath()
        for (let y = 0; y < res; y += step) {
          const h = (heightmap3DArray[y][x] - minH) * heightScale
          const px = canvas.width / 2 + (x - y) * scale * 0.866
          const py = canvas.height / 2 + (x + y) * scale * 0.5 - h + yOffset
          
          if (y === 0) {
            ctx.moveTo(px, py)
          } else {
            ctx.lineTo(px, py)
          }
        }
        ctx.stroke()
      }
      
    } catch (error) {
      console.error('2D Generation error:', error)
      return false
    } finally {
      setIsGenerating(false)
    }
    return true
  }, [pyodide, isReady, hurstIndex, resolution2D])

  // Raymarch rendering
  const setupRaymarch = useCallback(async () => {
    if (!canvasRaymarchRef.current || !heightmapRef.current) {
      console.log('Canvas or heightmap not ready')
      return
    }
    
    try {
      const canvas = canvasRaymarchRef.current
      const gl = canvas.getContext('webgl')
      if (!gl) {
        console.error('WebGL not supported')
        return
      }
      
      canvas.width = 800
      canvas.height = 600
      
      // Load shader
      const response = await fetch(`${import.meta.env.BASE_URL}mountain_raycast.glsl`)
      if (!response.ok) {
        console.error('Failed to load shader')
        return
      }
      const fragmentShaderSource = await response.text()
    
    const vertexShaderSource = `
      attribute vec2 position;
      void main() {
        gl_Position = vec4(position, 0.0, 1.0);
      }
    `
    
    // Compile shaders
    const vertexShader = gl.createShader(gl.VERTEX_SHADER)!
    gl.shaderSource(vertexShader, vertexShaderSource)
    gl.compileShader(vertexShader)
    
    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER)!
    gl.shaderSource(fragmentShader, fragmentShaderSource)
    gl.compileShader(fragmentShader)
    
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
      console.error('Fragment shader error:', gl.getShaderInfoLog(fragmentShader))
      return
    }
    
    // Create program
    const program = gl.createProgram()!
    gl.attachShader(program, vertexShader)
    gl.attachShader(program, fragmentShader)
    gl.linkProgram(program)
    
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(program))
      return
    }
    
    gl.useProgram(program)
    
    // Setup geometry
    const positions = new Float32Array([
      -1, -1,
      1, -1,
      -1, 1,
      1, 1
    ])
    
    const buffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW)
    
    const positionLocation = gl.getAttribLocation(program, 'position')
    gl.enableVertexAttribArray(positionLocation)
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0)
    
    // Setup texture
    const texture = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, heightmapRef.current)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT)
    
    // Get uniform locations
    const resolutionLocation = gl.getUniformLocation(program, 'u_resolution')
    const timeLocation = gl.getUniformLocation(program, 'u_time')
    const heightmapLocation = gl.getUniformLocation(program, 'u_heightmap')
    
    if (!resolutionLocation || !timeLocation || !heightmapLocation) {
      console.error('Failed to get uniform locations')
      return
    }
    
    gl.uniform2f(resolutionLocation, canvas.width, canvas.height)
    gl.uniform1i(heightmapLocation, 0)
    
    // Animation loop
    let startTime = Date.now()
    const render = () => {
      const time = (Date.now() - startTime) / 1000
      gl.uniform1f(timeLocation, time)
      
      gl.viewport(0, 0, canvas.width, canvas.height)
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
      
      animationFrameRef.current = requestAnimationFrame(render)
    }
    
    render()
    } catch (error) {
      console.error('Raymarch setup error:', error)
    }
  }, [])

  // Generate on tab change
  useEffect(() => {
    if (!isReady) return
    
    if (activeTab === '1d') {
      generate1D()
    } else if (activeTab === '2d') {
      generate2D()
    } else if (activeTab === 'raymarch') {
      // Raymarchタブでは常に新しいデータを生成
      generate2D(true).then((success) => {
        if (success && heightmapRef.current) {
          setTimeout(() => setupRaymarch(), 100)
        }
      }).catch(err => {
        console.error('Failed to generate 2D data for raymarch:', err)
      })
    }
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, isReady])

  // パラメータ変更時の自動再生成
  useEffect(() => {
    if (!isReady || isGenerating) return
    
    // 現在のタブに応じて再生成
    if (activeTab === '1d') {
      generate1D()
    } else if (activeTab === '2d') {
      generate2D()
    }
    // Raymarchタブでは生成ボタンでのみ再生成
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hurstIndex, resolution1D, resolution2D])

  const handleGenerate = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
    }
    
    if (activeTab === '1d') {
      generate1D()
    } else if (activeTab === '2d') {
      generate2D()
    } else if (activeTab === 'raymarch') {
      console.log('Starting raymarch generation...')
      generate2D(true).then((success) => {
        console.log('generate2D completed, success:', success, 'heightmapRef:', heightmapRef.current)
        if (success && heightmapRef.current) {
          console.log('Setting up raymarch...')
          setTimeout(() => setupRaymarch(), 100)
        } else {
          console.error('Failed to generate heightmap. Success:', success, 'HeightmapRef:', heightmapRef.current)
        }
      }).catch(err => {
        console.error('Raymarch generation error:', err)
      })
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🌄 Fractal Noise Generator</h1>
        <p className="subtitle">フーリエフィルタリングによるフラクタルノイズ生成</p>
      </header>

      {pyodideError && (
        <div className="error-message">
          Error: {pyodideError}
        </div>
      )}

      <div className="tabs">
        <button 
          className={activeTab === '1d' ? 'tab-button active' : 'tab-button'}
          onClick={() => setActiveTab('1d')}
        >
          1D Noise
        </button>
        <button 
          className={activeTab === '2d' ? 'tab-button active' : 'tab-button'}
          onClick={() => setActiveTab('2d')}
        >
          2D Mesh
        </button>
        <button 
          className={activeTab === 'raymarch' ? 'tab-button active' : 'tab-button'}
          onClick={() => setActiveTab('raymarch')}
        >
          Raymarch
        </button>
      </div>

      <div className="main-content">
        <aside className="sidebar">
          <div className="controls">
            <div className="controls-grid">
              <div className="control-group">
                <label>
                  Hurst Index (H): {hurstIndex.toFixed(2)}
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={hurstIndex}
                    onChange={(e) => setHurstIndex(parseFloat(e.target.value))}
                  />
                </label>
                <p className="hint">
                  H=0.5: ホワイトノイズ, H&lt;0.5: 荒い, H&gt;0.5: 滑らか
                </p>
              </div>

              {activeTab === '1d' && (
                <div className="control-group">
                  <label>
                    Resolution: {resolution1D}
                    <input
                      type="range"
                      min="128"
                      max="4096"
                      step="128"
                      value={resolution1D}
                      onChange={(e) => setResolution1D(parseInt(e.target.value))}
                    />
                  </label>
                </div>
              )}

              {activeTab === '2d' && (
                <div className="control-group">
                  <label>
                    Resolution: {resolution2D}x{resolution2D}
                    <input
                      type="range"
                      min="64"
                      max="512"
                      step="32"
                      value={resolution2D}
                      onChange={(e) => setResolution2D(parseInt(e.target.value))}
                    />
                  </label>
                </div>
              )}

              {activeTab === 'raymarch' && (
                <div className="control-group">
                  <p className="hint">
                    Resolution: 1024x1024 (Fixed for optimal quality)
                  </p>
                </div>
              )}
            </div>

            <button
              className="generate-button"
              onClick={handleGenerate}
              disabled={!isReady || isGenerating}
            >
              {isGenerating ? '生成中...' : '生成'}
            </button>
          </div>
        </aside>

        <div className="canvas-container">
          {activeTab === '1d' && (
            <canvas ref={canvas1DRef} className="noise-canvas" />
          )}
          {activeTab === '2d' && (
            <canvas ref={canvas2DRef} className="noise-canvas" />
          )}
          {activeTab === 'raymarch' && (
            <canvas ref={canvasRaymarchRef} className="noise-canvas" />
          )}
        </div>
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
