import { useEffect, useState } from 'react'
const BASE_URL = import.meta.env.BASE_URL

declare global {
  interface Window {
    loadPyodide: (config?: { indexURL?: string }) => Promise<PyodideInterface>
  }
}

export interface PyodideInterface {
  runPythonAsync: (code: string) => Promise<any>
  loadPackage: (packages: string[]) => Promise<void>
  globals: {
    get: (name: string) => any
  }
  FS: {
    writeFile: (path: string, data: string | Uint8Array) => void
    readFile: (path: string) => Uint8Array
  }
}

interface UsePyodideOptions {
  pythonFiles?: string[]  // 読み込むPythonファイルのパス（publicからの相対パス）
  packages?: string[]     // 追加で読み込むパッケージ
}

export function usePyodide(pythonFiles: string[] = [], options: UsePyodideOptions = {}) {
  const [pyodide, setPyodide] = useState<PyodideInterface | null>(null)
  const [isReady, setIsReady] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const { packages = [] } = options

  useEffect(() => {
    let cancelled = false

    async function loadPyodideAndPackages() {
      try {
        console.log('Loading Pyodide...')
        
        // CDNからPyodideスクリプトを読み込み
        if (!window.loadPyodide) {
          const script = document.createElement('script')
          script.src = 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js'
          script.async = true
          await new Promise((resolve, reject) => {
            script.onload = resolve
            script.onerror = reject
            document.head.appendChild(script)
          })
        }
        
        const pyodideInstance = await window.loadPyodide({
          indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/'
        })

        if (cancelled) return

        // 基本パッケージ + 追加パッケージを読み込み
        const allPackages = ['numpy', 'scipy', ...packages]
        console.log(`Loading packages: ${allPackages.join(', ')}...`)
        await pyodideInstance.loadPackage(allPackages)

        if (cancelled) return

        // Pythonファイルを読み込み
        for (const filePath of pythonFiles) {
          console.log(`Loading Python module: ${filePath}...`)
          const response = await fetch(`${BASE_URL}${filePath}`)
          if (!response.ok) {
            throw new Error(`Failed to fetch ${filePath}: ${response.statusText}`)
          }
          const pythonCode = await response.text()
          
          // ファイル名を取得してPyodideのファイルシステムに書き込み
          const fileName = filePath.split('/').pop() || filePath
          pyodideInstance.FS.writeFile(`/${fileName}`, pythonCode)
        }
        
        // モジュールをインポート可能にする
        await pyodideInstance.runPythonAsync(`
import sys
if '/' not in sys.path:
    sys.path.append('/')
`)

        if (cancelled) return

        setPyodide(pyodideInstance)
        setIsReady(true)
        console.log('Pyodide ready!')
      } catch (err) {
        if (!cancelled) {
          console.error('Failed to load Pyodide:', err)
          setError(err instanceof Error ? err.message : String(err))
        }
      }
    }

    loadPyodideAndPackages()

    return () => {
      cancelled = true
    }
  }, [pythonFiles.join(','), packages.join(',')])

  return { pyodide, isReady, error }
}
