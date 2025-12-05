import './App.css'

interface AppInfo {
  id: string
  title: string
  description: string
  path: string
  icon: string
}

const BASE_URL = import.meta.env.BASE_URL

const apps: AppInfo[] = [
  {
    id: 'spectral-mesh',
    title: 'Spectral Mesh Filter',
    description: '3DメッシュのスペクトルＩ分解とローパスフィルター処理。OBJファイルを読み込んで、メッシュの高周波成分を除去します。',
    path: `${BASE_URL}spectral-mesh/`,
    icon: '🔬'
  },
  {
    id: 'fractal-music',
    title: 'Fractal Music Generator',
    description: 'フラクタル理論（1/fノイズ）に基づく自動作曲。Hurst指数を調整して、変化に富んだメロディや滑らかなメロディを生成。',
    path: `${BASE_URL}fractal-music/`,
    icon: '🎵'
  },
  {
    id: 'hele-shaw',
    title: 'Hele-Shaw Flow Simulation',
    description: '基本解近似解法（MFS）によるヘレショウ流れの数値シミュレーション。表面張力による曲線の時間発展を可視化。',
    path: `${BASE_URL}hele-shaw/`,
    icon: '🌊'
  },
  {
    id: 'hele-shaw-gap',
    title: 'Hele-Shaw Gap Rising Flow',
    description: '基本解近似解法（MFS）による雙間上昇ヘレショウ流れ。時間変化する雙間幅における気泡の不安定性を可視化。',
    path: `${BASE_URL}hele-shaw-gap/`,
    icon: '🫧'
  },
  {
    id: 'viscous-fingering',
    title: 'Viscous Fingering',
    description: 'Saffman-Taylor不安定性による指状パターン形成。粘性流体の界面不安定性を可視化。',
    path: `${BASE_URL}viscous-fingering/`,
    icon: '🖐️'
  },
  // 新しいアプリはここに追加
]

function App() {
  return (
    <div className="portal-container">
      <header className="portal-header">
        <h1>Pyodide Web Apps</h1>
        <p>ブラウザ上で動作するPython科学計算アプリケーション</p>
      </header>
      
      <main className="apps-grid">
        {apps.map(app => (
          <a key={app.id} href={app.path} className="app-card">
            <div className="app-icon">{app.icon}</div>
            <h2>{app.title}</h2>
            <p>{app.description}</p>
          </a>
        ))}
      </main>
      
      <footer className="portal-footer">
        <p>Powered by <a href="https://pyodide.org/" target="_blank" rel="noopener noreferrer">Pyodide</a></p>
      </footer>
    </div>
  )
}

export default App
