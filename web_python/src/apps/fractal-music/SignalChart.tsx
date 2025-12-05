import { useEffect, useRef } from 'react'
import { Chart, registerables } from 'chart.js'

Chart.register(...registerables)

interface SignalChartProps {
  trackSignals: number[][]
  trackNames: string[]
  hurstIndex: number
}

// トラックごとの色
const trackColors = [
  { border: '#646cff', background: 'rgba(100, 108, 255, 0.2)' },  // Melody - 青紫
  { border: '#ff6b6b', background: 'rgba(255, 107, 107, 0.2)' },  // Bass - 赤
  { border: '#4ecdc4', background: 'rgba(78, 205, 196, 0.2)' },   // Chords - ターコイズ
]

function SignalChart({ trackSignals, trackNames, hurstIndex }: SignalChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartRef = useRef<Chart | null>(null)

  useEffect(() => {
    if (!canvasRef.current || trackSignals.length === 0) return

    // 既存のチャートを破棄
    if (chartRef.current) {
      chartRef.current.destroy()
    }

    const ctx = canvasRef.current.getContext('2d')
    if (!ctx) return

    // 各トラックのデータセットを作成
    const datasets = trackSignals.map((signal, index) => ({
      label: `${trackNames[index] || `Track ${index + 1}`} (H=${hurstIndex.toFixed(2)})`,
      data: signal,
      borderColor: trackColors[index % trackColors.length].border,
      backgroundColor: trackColors[index % trackColors.length].background,
      borderWidth: 1.5,
      fill: true,
      pointRadius: 0,
      tension: 0.1
    }))

    // 最長の信号の長さを取得
    const maxLength = Math.max(...trackSignals.map(s => s.length))

    chartRef.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: Array.from({ length: maxLength }, (_, i) => i.toString()),
        datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            labels: {
              color: '#ccc'
            }
          },
          title: {
            display: true,
            text: 'フラクタル信号（各トラック）',
            color: '#ccc',
            font: {
              size: 16
            }
          }
        },
        scales: {
          x: {
            display: true,
            title: {
              display: true,
              text: 'サンプル番号',
              color: '#aaa'
            },
            ticks: {
              color: '#aaa',
              maxTicksLimit: 10
            },
            grid: {
              color: '#444'
            }
          },
          y: {
            display: true,
            title: {
              display: true,
              text: '振幅',
              color: '#aaa'
            },
            ticks: {
              color: '#aaa'
            },
            grid: {
              color: '#444'
            }
          }
        },
        interaction: {
          intersect: false,
          mode: 'index'
        }
      }
    })

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy()
        chartRef.current = null
      }
    }
  }, [trackSignals, trackNames, hurstIndex])

  return (
    <div className="chart-container">
      <canvas ref={canvasRef} />
    </div>
  )
}

export default SignalChart
