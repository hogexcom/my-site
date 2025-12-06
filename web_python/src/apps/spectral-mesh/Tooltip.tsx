import { useState, useRef, useEffect, type ReactNode } from 'react'

interface TooltipProps {
  content: ReactNode
  children?: ReactNode
}

export default function Tooltip({ content, children }: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [tooltipStyle, setTooltipStyle] = useState<React.CSSProperties>({})
  const triggerRef = useRef<HTMLButtonElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isVisible && triggerRef.current) {
      const triggerRect = triggerRef.current.getBoundingClientRect()
      const tooltipWidth = 280
      const tooltipHeight = tooltipRef.current?.offsetHeight || 200
      const padding = 10
      
      // 横位置: トリガーの中央を基準に、画面内に収まるよう調整
      let left = triggerRect.left + triggerRect.width / 2 - tooltipWidth / 2
      if (left < padding) {
        left = padding
      } else if (left + tooltipWidth > window.innerWidth - padding) {
        left = window.innerWidth - tooltipWidth - padding
      }
      
      // 縦位置: 下に収まらなければ上に表示
      let top: number
      if (triggerRect.bottom + tooltipHeight + padding > window.innerHeight) {
        top = triggerRect.top - tooltipHeight - 8
      } else {
        top = triggerRect.bottom + 8
      }
      
      setTooltipStyle({
        left: `${left}px`,
        top: `${top}px`,
      })
    }
  }, [isVisible])

  // クリック外で閉じる
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        isVisible &&
        triggerRef.current &&
        !triggerRef.current.contains(e.target as Node) &&
        tooltipRef.current &&
        !tooltipRef.current.contains(e.target as Node)
      ) {
        setIsVisible(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isVisible])

  return (
    <span className="tooltip-container">
      <button
        ref={triggerRef}
        className="tooltip-trigger"
        onClick={() => setIsVisible(!isVisible)}
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        type="button"
        aria-label="ヘルプ"
      >
        {children || '?'}
      </button>
      {isVisible && (
        <div
          ref={tooltipRef}
          className="tooltip-content"
          style={tooltipStyle}
        >
          {content}
        </div>
      )}
    </span>
  )
}
