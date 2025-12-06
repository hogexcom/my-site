---
title: "3Dメッシュのスペクトル分解とフィルタリング"
date: 2025-12-06
draft: false
tags: ["3D", "信号処理", "数学", "WebApp"]
categories: ["技術"]
summary: "3Dメッシュを周波数成分に分解し、音声のイコライザーのようにフィルタリングできるWebアプリを作りました。"
---

## はじめに

音声や画像には「周波数」という概念があり、ローパスフィルターやハイパスフィルターで特定の成分を強調・除去できます。では、3Dモデルにも同じことができるのでしょうか？

答えは **Yes** です。

3Dメッシュも周波数成分に分解でき、「低周波 = なめらかな大域的形状」「高周波 = 細かいディテール」として操作できます。この技術を **スペクトルメッシュ処理** と呼びます。

## デモアプリ

実際に試せるWebアプリを作りました：

👉 **[Spectral Mesh Filter](/my-site/python-apps/spectral-mesh/)**


### 機能

- **OBJ, STL, PLY, glTF, FBX形式**の3Dモデルを読み込み可能
- **スペクトル解析**: ラプラシアン行列の固有値分解
- **フィルターカーブ**: 各周波数成分の強度をカーブで指定
- **基底関数の可視化**: 固有ベクトル（調和関数）をメッシュ上に表示

## 理論的背景

このアプリは以下の論文に基づいています：

> **Spectral Geometry Processing with Manifold Harmonics**  
> Bruno Vallet and Bruno Lévy  
> *Computer Graphics Forum (Eurographics 2008)*

### Manifold Harmonics Transform (MHT)

MHTは、3Dメッシュ上で定義されたフーリエ変換の拡張です。

1Dの信号 $f(t)$ に対するフーリエ変換が
$$\hat{f}(\omega) = \int f(t) e^{-i\omega t} dt$$
であるように、メッシュ上の関数 $f$ に対するMHTは
$$\hat{f}_k = \langle f, H_k \rangle = \sum_i f_i \cdot H_k(v_i)$$
と定義されます。

ここで $H_k$ はラプラシアン行列の固有ベクトル（**調和関数**）です。

### ラプラシアン行列

メッシュのラプラシアン行列 $\Delta$ は、頂点間の関係を表す行列です。論文では **DEC (Discrete Exterior Calculus)** に基づく定式化を使用：

$$\bar{\Delta} = \star_0^{-1} \cdot \Delta \cdot \star_0$$

この対称化されたラプラシアンの固有値分解により、周波数（固有値）と基底関数（固有ベクトル）が得られます。

### 周波数フィルタリング

固有値 $\lambda_k$ に対応する周波数は $\omega_k = \sqrt{\lambda_k}$ です。

フィルター関数 $h(\omega)$ を適用した再構成は：
$$f'(v_i) = \sum_k h(\omega_k) \cdot \hat{f}_k \cdot H_k(v_i)$$

- **ローパスフィルター**: 高周波をカット → なめらかに
- **ハイパスフィルター**: 低周波を抑制 → ディテールを強調
- **カスタムカーブ**: 任意の周波数応答を設計

## 実装について

### 技術スタック

- **Pyodide**: ブラウザ上でPython (NumPy, SciPy) を実行
- **Three.js / React Three Fiber**: 3D描画
- **TypeScript + React**: UI

### 計算の流れ

1. **ラプラシアン行列の構築**: コタンジェント重みを使用
2. **固有値分解**: Shift-Invert Lanczos法で低周波側から計算
3. **MHT**: 頂点座標をスペクトル係数に変換
4. **フィルタリング**: カーブに従って係数を調整
5. **逆MHT**: スペクトル係数を頂点座標に戻す

### 残差の扱い

全固有対を計算するのは計算量的に非現実的なため、計算した固有対で表現できない「残差」が生じます。この残差は超高周波成分として、フィルターカーブの平均値を適用しています。

## 応用例

- **メッシュのスムージング**: ノイズ除去
- **ディテールの強調**: 彫刻的な表現
- **形状圧縮**: 低周波成分のみ保存
- **形状モーフィング**: スペクトル空間での補間

## まとめ

スペクトルメッシュ処理は、音声処理のイコライザーのような直感的な操作を3Dモデルに適用できる強力な手法です。ぜひデモアプリで試してみてください！

## 参考文献

1. Vallet, B., & Lévy, B. (2008). Spectral Geometry Processing with Manifold Harmonics. *Computer Graphics Forum*, 27(2), 251-260.
2. Taubin, G. (1995). A signal processing approach to fair surface design. *SIGGRAPH '95*.
3. Desbrun, M., Meyer, M., Schröder, P., & Barr, A. H. (1999). Implicit fairing of irregular meshes using diffusion and curvature flow. *SIGGRAPH '99*.
