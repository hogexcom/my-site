import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

// https://vite.dev/config/
export default defineConfig({
  base: '/my-site/python-apps/',
  plugins: [react()],
  define: {
    'global': 'globalThis',
  },
  optimizeDeps: {
    esbuildOptions: {
      define: {
        global: 'globalThis'
      }
    }
  },
  build: {
    rollupOptions: {
      input: {
        // メインページ（アプリ一覧）
        main: resolve(__dirname, 'index.html'),
        // 個別アプリページ
        'spectral-mesh': resolve(__dirname, 'spectral-mesh/index.html'),
        'escherization': resolve(__dirname, 'escherization/index.html'),
        'fractal-music': resolve(__dirname, 'fractal-music/index.html'),
        'fractal-noise': resolve(__dirname, 'fractal-noise/index.html'),
        'fractal-planet': resolve(__dirname, 'fractal-planet/index.html'),
        'hele-shaw': resolve(__dirname, 'hele-shaw/index.html'),
        'hele-shaw-gap': resolve(__dirname, 'hele-shaw-gap/index.html'),
        'viscous-fingering': resolve(__dirname, 'viscous-fingering/index.html'),
      },
      output: {
        // コード分割設定
        manualChunks: {
          // React系の共通ライブラリ
          'vendor-react': ['react', 'react-dom'],
          // Three.js系
          'vendor-three': ['three', '@react-three/fiber', '@react-three/drei'],
          // Chart.js
          'vendor-chart': ['chart.js'],
          // Pyodide（各アプリで必要時に動的ロード）
        }
      }
    }
  }
})
