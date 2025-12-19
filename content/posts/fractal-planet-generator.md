+++
date = '2025-12-19T00:00:00+09:00'
draft = false
title = '断層線アルゴリズムによるフラクタル惑星生成'
tags = ['フラクタル', '数値シミュレーション', 'Python', 'Three.js', 'WebGL', 'webapp']
categories = ['技術', '数学']
+++

## フラクタル惑星生成アプリ

ランダムな断層線（大円）を累積的に配置することで、リアルな地形を持つフラクタル惑星を生成するWebアプリケーションを作成しました。

**[アプリを試す →](/my-site/python-apps/fractal-planet/)**

## 断層線アルゴリズムとは

断層線アルゴリズム（Fault Line Algorithm）は、地形生成の古典的な手法の一つで、地質学的な断層運動を模倣することでフラクタル的な地形を作り出します。

### 基本原理

球面上での断層線アルゴリズムは以下のステップで動作します：

1. **ランダムな大円の生成**
   - 球の中心を通る平面をランダムに生成
   - この平面と球面の交線が「断層線」となる

2. **地形の変位**
   - 断層線の両側で高度を上下させる
   - 片側を`+d`、反対側を`-d`だけ変位

3. **反復処理**
   - このプロセスを数百回繰り返す
   - 各断層の変位量は一定または減衰

4. **フラクタル地形の生成**
   - 多数の断層が重なることで複雑な地形が形成される
   - 自然な山脈、平野、海洋が現れる

### 数学的定式化

球面上の点 $\mathbf{r} = (x, y, z)$（単位球上）と、ランダムな単位法線ベクトル $\mathbf{n} = (n_x, n_y, n_z)$ について：

大円による平面の方程式：
$$
\mathbf{n} \cdot \mathbf{r} = 0
$$

点 $\mathbf{r}$ がどちら側にあるかは内積の符号で判定：
$$
\text{side}(\mathbf{r}) = \text{sign}(\mathbf{n} \cdot \mathbf{r})
$$

$i$ 回目の断層による高度変化：
$$
h_i(\mathbf{r}) = d_i \cdot \text{side}_i(\mathbf{r})
$$

総合的な高度：
$$
H(\mathbf{r}) = \sum_{i=1}^{N} h_i(\mathbf{r})
$$

### 座標系の変換

#### 球面座標から直交座標への変換

経度 $\lambda \in [-\pi, \pi]$、緯度 $\phi \in [-\pi/2, \pi/2]$ として：

$$
\begin{aligned}
x &= \cos\phi \cos\lambda \\
y &= \cos\phi \sin\lambda \\
z &= \sin\phi
\end{aligned}
$$

#### Plate Carrée投影

球面を2次元平面に展開する最もシンプルな投影法：
- 横軸：経度 $\lambda$（-180° ～ +180°）
- 縦軸：緯度 $\phi$（-90° ～ +90°）

この投影は極付近で歪みが大きいですが、実装が容易で全球を一覧できます。

## アプリの特徴

### 1. 効率的な累積計算

従来の方法では、100断層、200断層、300断層...と生成するたびに0からやり直していましたが、このアプリでは：

```python
def generate_multiple_planets(width, height, fault_counts=[100, 200, 300, 400, 500]):
    textures = []
    planet = FractalPlanet(width=width, height=height)
    
    prev_count = 0
    for num_faults in fault_counts:
        # 追加の断層のみを生成
        additional_faults = num_faults - prev_count
        planet.generate_terrain(num_faults=additional_faults)
        
        # 現在の状態をテクスチャとして保存
        texture = planet.get_texture_array()
        textures.append(texture)
        
        prev_count = num_faults
    
    return textures
```

この方法により、計算時間を大幅に短縮できます。

### 2. Earth-likeカラーマッピング

高度に応じて地球らしい配色を適用：

| 高度範囲 | 色 | 地形 |
|---------|-----|------|
| 0.0 - 0.3 | 深い青 | 深海 |
| 0.3 - 0.4 | 青 | 海洋 |
| 0.4 - 0.45 | 水色 | 浅瀬 |
| 0.45 - 0.5 | ベージュ | 海岸 |
| 0.5 - 0.6 | 緑 | 低地 |
| 0.6 - 0.7 | 明るい緑 | 平野 |
| 0.7 - 0.8 | 茶色 | 丘陵 |
| 0.8 - 0.9 | 灰色 | 山岳 |
| 0.9 - 1.0 | 白 | 雪山 |

### 3. Three.jsによる3D可視化

- **リアルなライティング**
  - 太陽光（Directional Light）
  - アンビエントライト（全体照明）
  - 半球ライト（空からの間接光）
  - リムライト（惑星の縁を強調）

- **インタラクティブな操作**
  - OrbitControlsによる自由な視点移動
  - 自動回転アニメーション
  - ズームイン/アウト

- **宇宙空間の演出**
  - 5000個のランダムな星
  - 太陽とグローエフェクト
  - 暗い背景と美しいコントラスト

### 4. UIの工夫

- **左右分割レイアウト**
  - 左：断層数別の2D地図リスト
  - 右：500断層の3D惑星ビュー

- **クリックで拡大表示**
  - 地図をクリックすると高解像度モーダル表示
  - 詳細な地形パターンを確認可能

## 技術スタック

### フロントエンド
- **React + TypeScript**：UI構築
- **Three.js**：3Dレンダリング
- **Vite**：高速ビルドツール

### 計算エンジン
- **Pyodide**：ブラウザ上でPythonを実行
- **NumPy**：高速な数値計算
- **科学計算ライブラリ**：効率的な配列操作

### テクスチャ生成
- 解像度：1024×512ピクセル
- カラーフォーマット：RGBA（32bit）
- 球面座標からの変換

## フラクタル地形生成の応用

### ゲーム開発
- プロシージャル惑星生成
- 無限に続くオープンワールド
- ランダムマップジェネレータ

### CG・映像制作
- SF映画の未知の惑星
- アニメーションの背景
- 概念アート

### 科学シミュレーション
- 地形モデリング
- 地質学的プロセスの研究
- 惑星形成シミュレーション

### データ可視化
- 多次元データの表現
- 統計的パターンの視覚化

## 今後の拡張案

1. **パラメータ調整UI**
   - 変位量の調整
   - 減衰率の設定
   - リアルタイムプレビュー

2. **追加の地形アルゴリズム**
   - Perlinノイズ
   - Simplex noise
   - Diamond-square algorithm

3. **物理ベースレンダリング**
   - 大気散乱シミュレーション
   - 雲のレイヤー
   - 海面の反射

4. **エクスポート機能**
   - OBJ形式で3Dメッシュ保存
   - PNG形式でテクスチャ保存
   - 高解像度レンダリング

## 参考文献

1. Peitgen, H.-O., & Saupe, D. (Eds.). (1988). *The Science of Fractal Images*. Springer-Verlag.

2. Musgrave, F. K., Kolb, C. E., & Mace, R. S. (1989). "The synthesis and rendering of eroded fractal terrains." *ACM SIGGRAPH Computer Graphics*, 23(3), 41-50.

3. Fournier, A., Fussell, D., & Carpenter, L. (1982). "Computer rendering of stochastic models." *Communications of the ACM*, 25(6), 371-384.

4. Miller, G. S. P. (1986). "The definition and rendering of terrain maps." *ACM SIGGRAPH Computer Graphics*, 20(4), 39-48.

5. Ebert, D. S., Musgrave, F. K., Peachey, D., Perlin, K., & Worley, S. (2002). *Texturing & Modeling: A Procedural Approach* (3rd ed.). Morgan Kaufmann.

## まとめ

断層線アルゴリズムは、シンプルながらも効果的なフラクタル地形生成手法です。このアプリでは、Pythonの数値計算とThree.jsの3D可視化を組み合わせることで、ブラウザ上でインタラクティブに惑星を生成・観察できるようにしました。

累積的な計算手法により、異なる複雑度の地形を効率的に比較でき、フラクタル構造の形成過程を視覚的に理解することができます。

ぜひアプリで自分だけの惑星を生成してみてください！
