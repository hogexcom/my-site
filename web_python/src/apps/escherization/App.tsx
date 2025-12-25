import { useEffect, useState } from 'react'
import './App.css'
import { ContourPrepTestPage } from './pages/ContourPrepTestPage'
import { EscherizationPipelinePage } from './pages/EscherizationPipelinePage'
import { ArapInteractiveTestPage } from './pages/ArapInteractiveTestPage'
import { TriangulationTestPage } from './pages/TriangulationTestPage'
import { EscherizationMainPage } from './pages/EscherizationMainPage'

type PageId =
  | 'escherization-main'
  | 'home'
  | 'contour-prep-test'
  | 'triangulation-test'
  | 'arap-interactive-test'
  | 'escherization-pipeline'

function normalizeHashToPageId(hash: string): PageId {
  const h = hash.replace(/^#/, '')
  if (h === 'escherization-main') return 'escherization-main'
  if (h === 'contour-prep-test') return 'contour-prep-test'
  if (h === 'triangulation-test') return 'triangulation-test'
  if (h === 'arap-interactive-test') return 'arap-interactive-test'
  if (h === 'escherization-pipeline') return 'escherization-pipeline'
  return 'home'
}

function App() {
  const [pageId, setPageId] = useState<PageId>(() => normalizeHashToPageId(window.location.hash))

  useEffect(() => {
    const onHashChange = () => setPageId(normalizeHashToPageId(window.location.hash))
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  return (
    <div className="escher-shell">
      <div className="escher-layout">
        <main className="escher-main">
          {pageId === 'escherization-main' && <EscherizationMainPage />}
          {pageId === 'home' && <EscherizationMainPage />}
          {pageId === 'escherization-pipeline' && <EscherizationPipelinePage />}
          {pageId === 'contour-prep-test' && <ContourPrepTestPage />}
          {pageId === 'triangulation-test' && <TriangulationTestPage />}
          {pageId === 'arap-interactive-test' && <ArapInteractiveTestPage />}
        </main>
      </div>
    </div>
  )
}

export default App
