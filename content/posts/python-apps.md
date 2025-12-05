+++
date = '2025-12-05T12:20:14+09:00'
draft = false
title = 'Pyodide Web Apps - ブラウザで動くPython科学計算アプリ'
tags = ['Python', 'Pyodide', 'Webアプリ', '科学計算']
categories = ['プロジェクト']
+++

## Pyodide Web Apps

Pythonの科学計算ライブラリをブラウザ上で実行できるアプリケーション集を作成しました。

**[→ アプリ一覧へ](/my-site/python-apps/)**

## 特徴

- **インストール不要**: ブラウザだけでPythonが動きます
- **Pyodide採用**: WebAssemblyでコンパイルされたPythonを使用
- **リアルタイム処理**: パラメータを調整してすぐに結果を確認

## 収録アプリ

### 🎵 Fractal Music Generator

フラクタル理論（1/fノイズ）に基づく自動作曲アプリ。Hurst指数を調整して、多様なメロディを生成できます。

### 🌊 Hele-Shaw Flow Simulation

基本解近似解法（MFS）によるヘレショウ流れの数値シミュレーション。表面張力による曲線の時間発展を可視化します。

### 🫧 Hele-Shaw Gap Rising Flow

時間変化する隙間幅におけるヘレショウ流れのシミュレーション。気泡の不安定性を可視化します。

### 🖐️ Viscous Fingering

Saffman-Taylor不安定性による指状パターン形成。粘性流体の界面不安定性を可視化します。

### 🔬 Spectral Mesh Filter

3Dメッシュのスペクトル分解とローパスフィルター処理。OBJファイルを読み込んでメッシュの高周波成分を除去できます。

## 技術スタック

- **Pyodide** - WebAssembly上で動作するPython
- **React** - UIフレームワーク
- **Vite** - 高速なビルドツール
- **Three.js** - 3Dグラフィックス
- **NumPy / SciPy** - 科学計算

**[→ アプリ一覧を見る](/my-site/python-apps/)**
