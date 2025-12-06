import { useState, useRef, useCallback, useEffect } from 'react'

export interface ControlPoint {
  x: number  // 0-1 (固有値インデックスの正規化)
  y: number  // 0-2 (フィルター値)
}

interface CurveEditorProps {
  points: ControlPoint[]
  onChange: (points: ControlPoint[]) => void
  maxEigenpairs: number
  disabled?: boolean
}

// 制御点間を線形補間してカーブの値を取得
export function evaluateCurve(points: ControlPoint[], x: number): number {
  if (points.length === 0) return 1
  if (points.length === 1) return points[0].y
  
  // xでソート済みの点を前提
  const sorted = [...points].sort((a, b) => a.x - b.x)
  
  // 範囲外
  if (x <= sorted[0].x) return sorted[0].y
  if (x >= sorted[sorted.length - 1].x) return sorted[sorted.length - 1].y
  
  // 補間
  for (let i = 0; i < sorted.length - 1; i++) {
    if (x >= sorted[i].x && x <= sorted[i + 1].x) {
      const t = (x - sorted[i].x) / (sorted[i + 1].x - sorted[i].x)
      return sorted[i].y + t * (sorted[i + 1].y - sorted[i].y)
    }
  }
  
  return 1
}

// カーブの平均値を計算（0-1の範囲で積分）
export function calculateCurveAverage(points: ControlPoint[], samples: number = 100): number {
  let sum = 0
  for (let i = 0; i <= samples; i++) {
    const x = i / samples
    sum += evaluateCurve(points, x)
  }
  return sum / (samples + 1)
}

export default function CurveEditor({ points, onChange, maxEigenpairs, disabled }: CurveEditorProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [draggingIndex, setDraggingIndex] = useState<number | null>(null)
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)
  
  const width = 280
  const height = 150
  const padding = { top: 20, right: 20, bottom: 30, left: 35 }
  const plotWidth = width - padding.left - padding.right
  const plotHeight = height - padding.top - padding.bottom
  
  // 座標変換
  const toScreenX = (x: number) => padding.left + x * plotWidth
  const toScreenY = (y: number) => padding.top + (1 - y / 2) * plotHeight
  const fromScreenX = (sx: number) => Math.max(0, Math.min(1, (sx - padding.left) / plotWidth))
  const fromScreenY = (sy: number) => Math.max(0, Math.min(2, 2 * (1 - (sy - padding.top) / plotHeight)))
  
  // カーブのパスを生成
  const generatePath = useCallback(() => {
    if (points.length === 0) return ''
    
    const sorted = [...points].sort((a, b) => a.x - b.x)
    const pathPoints: string[] = []
    
    // 0からカーブの開始点まで
    pathPoints.push(`M ${toScreenX(0)} ${toScreenY(sorted[0].y)}`)
    pathPoints.push(`L ${toScreenX(sorted[0].x)} ${toScreenY(sorted[0].y)}`)
    
    // 制御点間を線形補間
    for (let i = 1; i < sorted.length; i++) {
      pathPoints.push(`L ${toScreenX(sorted[i].x)} ${toScreenY(sorted[i].y)}`)
    }
    
    // カーブの終点から1まで
    pathPoints.push(`L ${toScreenX(1)} ${toScreenY(sorted[sorted.length - 1].y)}`)
    
    return pathPoints.join(' ')
  }, [points])
  
  // マウスイベント
  const handleMouseDown = useCallback((e: React.MouseEvent, index: number) => {
    if (disabled) return
    e.preventDefault()
    setDraggingIndex(index)
  }, [disabled])
  
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (draggingIndex === null || disabled) return
    
    const svg = svgRef.current
    if (!svg) return
    
    const rect = svg.getBoundingClientRect()
    const x = fromScreenX(e.clientX - rect.left)
    const y = fromScreenY(e.clientY - rect.top)
    
    const newPoints = [...points]
    newPoints[draggingIndex] = { x, y }
    onChange(newPoints)
  }, [draggingIndex, points, onChange, disabled])
  
  const handleMouseUp = useCallback(() => {
    setDraggingIndex(null)
  }, [])
  
  // ダブルクリックで点を追加/削除
  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    if (disabled) return
    
    const svg = svgRef.current
    if (!svg) return
    
    const rect = svg.getBoundingClientRect()
    const x = fromScreenX(e.clientX - rect.left)
    const y = fromScreenY(e.clientY - rect.top)
    
    // 既存の点の近くをクリックした場合は削除（最低2点は残す）
    const clickThreshold = 0.05
    const nearIndex = points.findIndex(p => 
      Math.abs(p.x - x) < clickThreshold && Math.abs(p.y - y) < 0.2
    )
    
    if (nearIndex !== -1 && points.length > 2) {
      const newPoints = points.filter((_, i) => i !== nearIndex)
      onChange(newPoints)
    } else {
      // 新しい点を追加
      onChange([...points, { x, y }])
    }
  }, [points, onChange, disabled])
  
  // グローバルマウスアップイベント
  useEffect(() => {
    const handleGlobalMouseUp = () => setDraggingIndex(null)
    window.addEventListener('mouseup', handleGlobalMouseUp)
    return () => window.removeEventListener('mouseup', handleGlobalMouseUp)
  }, [])
  
  // Y軸ラベル
  const yLabels = [0, 0.5, 1, 1.5, 2]
  
  return (
    <div className="curve-editor">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onDoubleClick={handleDoubleClick}
        style={{ cursor: disabled ? 'not-allowed' : draggingIndex !== null ? 'grabbing' : 'crosshair' }}
      >
        {/* 背景 */}
        <rect
          x={padding.left}
          y={padding.top}
          width={plotWidth}
          height={plotHeight}
          fill="#1a1a1a"
          stroke="#444"
        />
        
        {/* Y=1のガイドライン */}
        <line
          x1={padding.left}
          y1={toScreenY(1)}
          x2={padding.left + plotWidth}
          y2={toScreenY(1)}
          stroke="#666"
          strokeDasharray="4,4"
        />
        
        {/* グリッド */}
        {[0.25, 0.5, 0.75].map(x => (
          <line
            key={`grid-x-${x}`}
            x1={toScreenX(x)}
            y1={padding.top}
            x2={toScreenX(x)}
            y2={padding.top + plotHeight}
            stroke="#333"
          />
        ))}
        {[0.5, 1.5].map(y => (
          <line
            key={`grid-y-${y}`}
            x1={padding.left}
            y1={toScreenY(y)}
            x2={padding.left + plotWidth}
            y2={toScreenY(y)}
            stroke="#333"
          />
        ))}
        
        {/* カーブ */}
        <path
          d={generatePath()}
          fill="none"
          stroke="#646cff"
          strokeWidth={2}
        />
        
        {/* 制御点 */}
        {points.map((point, index) => (
          <g key={index}>
            <circle
              cx={toScreenX(point.x)}
              cy={toScreenY(point.y)}
              r={hoveredIndex === index ? 8 : 6}
              fill={draggingIndex === index ? '#ff6b6b' : '#646cff'}
              stroke="#fff"
              strokeWidth={2}
              style={{ cursor: disabled ? 'not-allowed' : 'grab' }}
              onMouseDown={(e) => handleMouseDown(e, index)}
              onMouseEnter={() => setHoveredIndex(index)}
              onMouseLeave={() => setHoveredIndex(null)}
            />
          </g>
        ))}
        
        {/* Y軸ラベル */}
        {yLabels.map(y => (
          <text
            key={`label-y-${y}`}
            x={padding.left - 5}
            y={toScreenY(y)}
            textAnchor="end"
            dominantBaseline="middle"
            fill="#999"
            fontSize={10}
          >
            {y}
          </text>
        ))}
        
        {/* X軸ラベル */}
        <text
          x={padding.left}
          y={height - 5}
          textAnchor="start"
          fill="#999"
          fontSize={10}
        >
          1
        </text>
        <text
          x={padding.left + plotWidth}
          y={height - 5}
          textAnchor="end"
          fill="#999"
          fontSize={10}
        >
          {maxEigenpairs}
        </text>
        
        {/* 軸タイトル */}
        <text
          x={padding.left + plotWidth / 2}
          y={height - 5}
          textAnchor="middle"
          fill="#999"
          fontSize={10}
        >
          固有値インデックス
        </text>
        <text
          x={10}
          y={padding.top + plotHeight / 2}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="#999"
          fontSize={10}
          transform={`rotate(-90, 10, ${padding.top + plotHeight / 2})`}
        >
          フィルター値
        </text>
      </svg>
      <div className="curve-editor-hint">
        ダブルクリック: 点の追加/削除 | ドラッグ: 点の移動
      </div>
    </div>
  )
}
