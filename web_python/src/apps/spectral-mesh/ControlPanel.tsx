import { type ChangeEvent } from 'react'
import CurveEditor, { type ControlPoint } from './CurveEditor'
import Tooltip from './Tooltip'
import { SUPPORTED_EXTENSIONS } from '../../utils/meshLoaders'

interface SpectrumInfo {
  maxEigenpairs: number
  nyquistFrequency: number
}

interface ControlPanelProps {
  onFileSelect: (file: File) => void
  onAnalyze: () => void
  numEigenpairs: number
  onNumEigenpairsChange: (value: number) => void
  filterCurvePoints: ControlPoint[]
  onFilterCurveChange: (points: ControlPoint[]) => void
  selectedEigenvectorIndex: number
  onEigenvectorIndexChange: (value: number) => void
  isProcessing: boolean
  isAnalyzing: boolean
  isReady: boolean
  isLoadingDefault: boolean
  spectrumInfo: SpectrumInfo | null
  hasMesh: boolean
  analysisStatus: string
}

export default function ControlPanel({
  onFileSelect,
  onAnalyze,
  numEigenpairs,
  onNumEigenpairsChange,
  filterCurvePoints,
  onFilterCurveChange,
  selectedEigenvectorIndex,
  onEigenvectorIndexChange,
  isProcessing,
  isAnalyzing,
  isReady,
  isLoadingDefault,
  spectrumInfo,
  hasMesh,
  analysisStatus
}: ControlPanelProps) {
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      onFileSelect(file)
    }
  }

  return (
    <div className="control-panel">
      <h2>
        Spectral Mesh Filter
        <Tooltip content={
          <div>
            <strong>スペクトルメッシュフィルター</strong>
            <p>3Dメッシュを周波数成分に分解し、フィルタリングを行うツールです。</p>
            <p>メッシュの形状を「低周波（なめらかな形状）」と「高周波（細かいディテール）」に分けて操作できます。</p>
          </div>
        } />
      </h2>
      
      <div className="control-group">
        <label htmlFor="file-upload">
          Upload 3D Model
          <Tooltip content={
            <div>
              <strong>3Dモデルのアップロード</strong>
              <p>以下の形式に対応しています:</p>
              <ul>
                <li><strong>OBJ</strong> - Wavefront OBJ</li>
                <li><strong>STL</strong> - Stereolithography</li>
                <li><strong>PLY</strong> - Polygon File Format</li>
                <li><strong>glTF/GLB</strong> - GL Transmission Format</li>
                <li><strong>FBX</strong> - Autodesk FBX</li>
              </ul>
              <p>デフォルトではStanford Bunnyが読み込まれています。</p>
            </div>
          } />
        </label>
        <input
          id="file-upload"
          type="file"
          accept={SUPPORTED_EXTENSIONS}
          onChange={handleFileChange}
          disabled={isProcessing || isAnalyzing || !isReady}
        />
      </div>

      {/* Step 1: Analyze */}
      <div className="button-with-help">
        <button
          className="filter-button"
          onClick={onAnalyze}
          disabled={isProcessing || isAnalyzing || !isReady || !hasMesh}
        >
          {isAnalyzing ? 'Analyzing...' : 'Analyze Spectrum'}
        </button>
        <Tooltip content={
          <div>
            <strong>スペクトル解析</strong>
            <p>メッシュのラプラシアン行列を計算し、固有値分解を行います。</p>
            <p><strong>処理内容:</strong></p>
            <ol>
              <li>DECラプラシアン行列の構築</li>
              <li>固有値・固有ベクトルの計算</li>
              <li>フーリエ係数（MHT）の計算</li>
            </ol>
            <p>※ 頂点数が多いと数分かかる場合があります</p>
          </div>
        } />
      </div>

      {/* Step 2: スライダー（解析後のみ表示） */}
      {spectrumInfo && (
        <>
          <div className="control-group">
            <label htmlFor="eigenpairs">
              Eigenpairs: {numEigenpairs} / {spectrumInfo.maxEigenpairs}
              <Tooltip content={
                <div>
                  <strong>固有対数 (Eigenpairs)</strong>
                  <p>再構成に使用する固有値・固有ベクトルのペア数です。</p>
                  <ul>
                    <li><strong>少ない:</strong> 低周波成分のみ（なめらか）</li>
                    <li><strong>多い:</strong> より多くの周波数成分を含む（詳細）</li>
                  </ul>
                  <p>値を小さくするとローパスフィルター効果があります。</p>
                </div>
              } />
            </label>
            <input
              id="eigenpairs"
              type="range"
              min="10"
              max={spectrumInfo.maxEigenpairs}
              step="10"
              value={numEigenpairs}
              onChange={(e) => onNumEigenpairsChange(Number(e.target.value))}
              disabled={isProcessing || isAnalyzing}
            />
            <div className="range-labels">
              <span>10</span>
              <span>{spectrumInfo.maxEigenpairs}</span>
            </div>
          </div>

          <div className="control-group curve-editor-section">
            <label>
              フィルターカーブ (周波数応答)
              <Tooltip content={
                <div>
                  <strong>フィルターカーブ</strong>
                  <p>各周波数成分にどれだけの強度を適用するかを設定します。</p>
                  <ul>
                    <li><strong>横軸:</strong> 固有値インデックス（左=低周波、右=高周波）</li>
                    <li><strong>縦軸:</strong> フィルター値（0〜2）</li>
                  </ul>
                  <p><strong>操作方法:</strong></p>
                  <ul>
                    <li>ドラッグ: 制御点を移動</li>
                    <li>ダブルクリック: 点を追加/削除</li>
                  </ul>
                  <p><strong>フィルター値の意味:</strong></p>
                  <ul>
                    <li>1.0: 元のまま</li>
                    <li>0.0: 完全にカット</li>
                    <li>2.0: 2倍に増幅</li>
                  </ul>
                </div>
              } />
            </label>
            <CurveEditor
              points={filterCurvePoints}
              onChange={onFilterCurveChange}
              maxEigenpairs={spectrumInfo.maxEigenpairs}
              disabled={isProcessing || isAnalyzing}
            />
          </div>

          <div className="control-group eigenvector-section">
            <label htmlFor="eigenvector-index">
              基底関数表示: φ<sub>{selectedEigenvectorIndex}</sub>
              <Tooltip content={
                <div>
                  <strong>基底関数（固有ベクトル）の可視化</strong>
                  <p>選択した固有ベクトルをメッシュ上に色で表示します。</p>
                  <ul>
                    <li><strong>青:</strong> 負の値</li>
                    <li><strong>白:</strong> ゼロ付近</li>
                    <li><strong>赤:</strong> 正の値</li>
                  </ul>
                  <p>低いインデックスは低周波（大きな構造）、高いインデックスは高周波（細かい変動）を表します。</p>
                  <p>これらの基底関数の線形結合でメッシュ形状が表現されます。</p>
                </div>
              } />
            </label>
            <input
              id="eigenvector-index"
              type="range"
              min="1"
              max={spectrumInfo.maxEigenpairs - 1}
              step="1"
              value={selectedEigenvectorIndex}
              onChange={(e) => onEigenvectorIndexChange(Number(e.target.value))}
              disabled={isProcessing || isAnalyzing}
            />
            <div className="range-labels">
              <span>1 (低周波)</span>
              <span>{spectrumInfo.maxEigenpairs - 1} (高周波)</span>
            </div>
            <div className="color-legend">
              <span className="legend-negative">負</span>
              <span className="legend-positive">正</span>
            </div>
          </div>
        </>
      )}

      {!isReady && (
        <div className="status">Initializing Pyodide...</div>
      )}

      {isLoadingDefault && (
        <div className="status">Loading default model...</div>
      )}

      {/* 解析進捗表示 */}
      {analysisStatus && (
        <div className="status-box">
          <div className="status-label">Status:</div>
          <div className="status-text">{analysisStatus}</div>
          {isAnalyzing && <div className="status-spinner" />}
        </div>
      )}
    </div>
  )
}
