import { type ChangeEvent } from 'react'

interface ControlPanelProps {
  onFileSelect: (file: File) => void
  onFilter: () => void
  cutoffRatio: number
  onCutoffChange: (value: number) => void
  numEigenpairs: number
  onNumEigenpairsChange: (value: number) => void
  isProcessing: boolean
  isReady: boolean
}

export default function ControlPanel({
  onFileSelect,
  onFilter,
  cutoffRatio,
  onCutoffChange,
  numEigenpairs,
  onNumEigenpairsChange,
  isProcessing,
  isReady
}: ControlPanelProps) {
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      onFileSelect(file)
    }
  }

  return (
    <div className="control-panel">
      <h2>Spectral Mesh Filter</h2>
      
      <div className="control-group">
        <label htmlFor="file-upload">Upload OBJ File</label>
        <input
          id="file-upload"
          type="file"
          accept=".obj"
          onChange={handleFileChange}
          disabled={isProcessing || !isReady}
        />
      </div>

      <div className="control-group">
        <label htmlFor="eigenpairs">
          Eigenpairs: {numEigenpairs}
        </label>
        <input
          id="eigenpairs"
          type="range"
          min="10"
          max="300"
          step="10"
          value={numEigenpairs}
          onChange={(e) => onNumEigenpairsChange(Number(e.target.value))}
          disabled={isProcessing}
        />
        <div className="range-labels">
          <span>10</span>
          <span>300</span>
        </div>
      </div>

      <div className="control-group">
        <label htmlFor="cutoff">
          Cutoff Ratio: {cutoffRatio.toFixed(2)}
        </label>
        <input
          id="cutoff"
          type="range"
          min="0.1"
          max="0.9"
          step="0.05"
          value={cutoffRatio}
          onChange={(e) => onCutoffChange(Number(e.target.value))}
          disabled={isProcessing}
        />
        <div className="range-labels">
          <span>0.1</span>
          <span>0.9</span>
        </div>
      </div>

      <button
        className="filter-button"
        onClick={onFilter}
        disabled={isProcessing || !isReady}
      >
        {isProcessing ? 'Processing...' : 'Apply Lowpass Filter'}
      </button>

      {!isReady && (
        <div className="status">Initializing Pyodide...</div>
      )}
    </div>
  )
}
