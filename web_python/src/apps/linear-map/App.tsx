import './App.css'
import LinearMapVisualizer from '../../components/LinearMapVisualizer'

function App() {
  return (
    <div className="app-container">
      <header className="app-header">
        <div>
          <p className="eyebrow">Linear Map Playground</p>
          <h1>線型写像をドラッグで理解する</h1>
          <p className="subtitle">
            単位ベクトルの行き先を動かして、座標系がどう変形するかを観察します。
          </p>
        </div>
      </header>
      <LinearMapVisualizer />
    </div>
  )
}

export default App
