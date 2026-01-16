+++
date = '2026-01-16T18:00:00+09:00'
draft = false
title = 'Level Set Method + Navier-Stokes Simulation'
tags = ['Level Set Method', 'Navier-Stokes', 'Python', 'Simulation', 'Fluid Dynamics']
categories = ['Project', 'Math', 'Physics']
cover = { image = "/my-site/images/level-set-ns.png", alt = "Level Set Method and Navier-Stokes Simulation Catch Image" }
+++

## レベルセット法とナヴィエ・ストークス方程式

レベルセット法（Level Set Method）とナヴィエ・ストークス方程式（Navier-Stokes Equations）を組み合わせた流体シミュレーションアプリを作成しました。このシミュレーションでは、気液二相流のような、異なる性質を持つ流体間の界面の動きを追跡します。

**[→ シミュレーションを試す](/my-site/python-apps/level-set-ns/)**

### シミュレーションの仕組み

このシミュレーションは主に2つの要素で構成されています：

1.  **ナヴィエ・ストークス方程式**: 流体の速度場と圧力場を計算し、流体がどのように動くかを決定します。
2.  **レベルセット法**: 流体の界面（boundary）を、ある関数のレベルセット（等高線、通常は0レベル）として表現し、界面の形状変化やトポロジーの変化（分裂や結合など）を自然に扱います。

レベルセット法を用いることで、複雑に変形する界面を明示的なメッシュを使うことなく、固定グリッド上で高精度に追跡することが可能になります。

## 参考文献

このシミュレーションの実装にあたっては、以下の書籍を参考にしました。レベルセット法の基礎から応用、流体シミュレーションへの適用まで、詳細に解説されています。

*   **Level Set Methods and Dynamic Implicit Surfaces**
    *   Stanley Osher, Ronald Fedkiw
    *   Springer

## アプリについて

PythonとWASM技術を用いて、ブラウザ上で直接動作するシミュレーションとして実装されています。計算結果はリアルタイムに可視化され、パラメータを調整して流体の挙動の変化を観察することができます。

**[→ シミュレーションを試す](/my-site/python-apps/level-set-ns/)**
