"""
Spectral Mesh Processing for Pyodide/Browser

Pyodide環境で動作するスペクトル分解・フィルタリングモジュール
ファイルI/Oなし、すべてメモリ上で処理

主な機能:
1. DECラプラシアンの計算
2. Shift-Invert Lanczos法による固有値・固有ベクトル計算
3. ローパス/ハイパスフィルター
4. 残差（高周波成分）の計算と再注入

使用方法（Pyodideから）:
    processor = SpectralMeshProcessor(vertices, faces)
    processor.compute_spectrum(max_frequency=100)
    filtered_vertices = processor.apply_lowpass_filter(cutoff_freq=50)
"""

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh, splu


class SpectralMeshProcessor:
    """メッシュのスペクトル処理クラス（Pyodide用）"""
    
    def __init__(self, vertices, faces, dual_type='circumcentric'):
        """
        Parameters:
            vertices: np.ndarray (n_vertices, 3) - 頂点座標
            faces: np.ndarray (n_faces, 3) - 面の頂点インデックス（0-indexed）
            dual_type: str - 双対セルのタイプ ('barycentric' or 'circumcentric')
        """
        self.vertices = np.array(vertices, dtype=np.float64)
        self.faces = np.array(faces, dtype=np.int32)
        self.n_vertices = len(vertices)
        self.n_faces = len(faces)
        self.dual_type = dual_type
        
        # スペクトル分解の結果
        self.eigenvalues = None
        self.eigenvectors = None
        self.frequencies = None
        
        # フーリエ係数（MHT係数）
        self.fourier_X = None
        self.fourier_Y = None
        self.fourier_Z = None
        
        # 残差（高周波成分）
        self.residual_X = None
        self.residual_Y = None
        self.residual_Z = None
        
        # DECラプラシアン
        self.Delta_bar = None
        self.dual_areas = None
        
        # Nyquist周波数
        self.nyquist_frequency = None
    
    def get_memory_usage(self):
        """
        現在のメモリ使用量を推定（MB単位）
        
        Returns:
            dict - 各データ構造のメモリ使用量
        """
        usage = {}
        
        # 頂点データ
        usage['vertices'] = self.vertices.nbytes / (1024**2)
        usage['faces'] = self.faces.nbytes / (1024**2)
        
        # スペクトルデータ
        if self.eigenvalues is not None:
            usage['eigenvalues'] = self.eigenvalues.nbytes / (1024**2)
        if self.eigenvectors is not None:
            usage['eigenvectors'] = self.eigenvectors.nbytes / (1024**2)
        
        # フーリエ係数
        if self.fourier_X is not None:
            usage['fourier_coeffs'] = (self.fourier_X.nbytes + 
                                       self.fourier_Y.nbytes + 
                                       self.fourier_Z.nbytes) / (1024**2)
        
        # 残差
        if self.residual_X is not None:
            usage['residual'] = (self.residual_X.nbytes + 
                                self.residual_Y.nbytes + 
                                self.residual_Z.nbytes) / (1024**2)
        
        # 疎行列（推定値）
        if self.Delta_bar is not None:
            usage['laplacian_sparse'] = (self.Delta_bar.data.nbytes + 
                                        self.Delta_bar.indices.nbytes + 
                                        self.Delta_bar.indptr.nbytes) / (1024**2)
        
        usage['total'] = sum(usage.values())
        
        return usage
    
    def _compute_triangle_area(self, v0, v1, v2):
        """三角形の面積を計算"""
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        area = 0.5 * np.linalg.norm(cross)
        return area
    
    def _compute_dual_areas_barycentric(self):
        """Barycentric双対セル面積（各三角形の面積を3等分）"""
        dual_areas = np.zeros(self.n_vertices)
        
        for face in self.faces:
            v0, v1, v2 = face
            p0 = self.vertices[v0]
            p1 = self.vertices[v1]
            p2 = self.vertices[v2]
            
            area = self._compute_triangle_area(p0, p1, p2)
            dual_areas[v0] += area / 3.0
            dual_areas[v1] += area / 3.0
            dual_areas[v2] += area / 3.0
        
        return dual_areas
    
    def _compute_dual_areas_circumcentric(self):
        """Circumcentric双対セル面積（Voronoiセル）"""
        dual_areas = np.zeros(self.n_vertices)
        
        for face in self.faces:
            v0, v1, v2 = face
            p0 = self.vertices[v0]
            p1 = self.vertices[v1]
            p2 = self.vertices[v2]
            
            # エッジベクトル
            e0 = p1 - p0
            e1 = p2 - p1
            e2 = p0 - p2
            
            # エッジの長さの2乗
            l0_sq = np.dot(e0, e0)
            l1_sq = np.dot(e1, e1)
            l2_sq = np.dot(e2, e2)
            
            # 三角形の面積
            area = self._compute_triangle_area(p0, p1, p2)
            
            # コタンジェント
            # cot(angle) = dot(v1, v2) / ||cross(v1, v2)||
            cross01 = np.cross(e0, e1)
            cross12 = np.cross(e1, e2)
            cross20 = np.cross(e2, e0)
            
            cross_norm = np.linalg.norm(cross01) + 1e-12
            
            cot0 = -np.dot(e2, e0) / cross_norm
            cot1 = -np.dot(e0, e1) / cross_norm
            cot2 = -np.dot(e1, e2) / cross_norm
            
            # Voronoiセル面積 = (1/8) * Σ (l²) * cot(opposite angle)
            dual_areas[v0] += 0.125 * (l1_sq * cot1 + l2_sq * cot2)
            dual_areas[v1] += 0.125 * (l2_sq * cot2 + l0_sq * cot0)
            dual_areas[v2] += 0.125 * (l0_sq * cot0 + l1_sq * cot1)
        
        return dual_areas
    
    def _compute_dual_areas(self):
        """双対セル面積を計算（タイプに応じて）"""
        if self.dual_type == 'barycentric':
            return self._compute_dual_areas_barycentric()
        elif self.dual_type == 'circumcentric':
            return self._compute_dual_areas_circumcentric()
        else:
            raise ValueError(f"Unknown dual_type: {self.dual_type}")
    
    def compute_laplacian(self):
        """
        DECラプラシアン Δ̄ を計算（対称化版）
        
        Vallet & Lévy (2008) 式 (2):
            Δ̄ = ⋆₀⁻¹ · Δ · ⋆₀
        
        Returns:
            Delta_bar: scipy.sparse.csr_matrix - 対称化ラプラシアン
        """
        print("\nDECラプラシアン Δ̄ を計算中...")
        
        # 双対セル面積を計算
        self.dual_areas = self._compute_dual_areas()
        
        # コタンジェントラプラシアン Δ を構築
        # 行、列、データのリスト
        row_indices = []
        col_indices = []
        data_values = []
        
        # 各エッジのコタンジェント重みを計算
        edge_weights = {}  # {(i, j): weight}
        
        for face in self.faces:
            v0, v1, v2 = face
            p0 = self.vertices[v0]
            p1 = self.vertices[v1]
            p2 = self.vertices[v2]
            
            # エッジベクトル
            e0 = p1 - p0  # v0 -> v1
            e1 = p2 - p1  # v1 -> v2
            e2 = p0 - p2  # v2 -> v0
            
            # 各エッジに対するコタンジェント重み
            # エッジ (v0, v1) の対角の角度は v2
            # cot(angle at v2) = dot(e2, -e0) / ||cross(e2, -e0)||
            
            def add_cotangent_weight(vi, vj, opposite_vertex_pos):
                """エッジ (vi, vj) のコタンジェント重みを追加"""
                edge_vec = self.vertices[vj] - self.vertices[vi]
                v1_to_opposite = opposite_vertex_pos - self.vertices[vi]
                v2_to_opposite = opposite_vertex_pos - self.vertices[vj]
                
                cross_prod = np.cross(v1_to_opposite, v2_to_opposite)
                cross_norm = np.linalg.norm(cross_prod) + 1e-12
                
                dot_prod = np.dot(v1_to_opposite, v2_to_opposite)
                cot_weight = dot_prod / cross_norm
                
                # エッジの重みを累積（複数の面で共有される場合）
                edge_key = tuple(sorted([vi, vj]))
                if edge_key not in edge_weights:
                    edge_weights[edge_key] = 0.0
                edge_weights[edge_key] += cot_weight
            
            # 3つのエッジに重みを追加
            add_cotangent_weight(v0, v1, p2)
            add_cotangent_weight(v1, v2, p0)
            add_cotangent_weight(v2, v0, p1)
        
        # 数値安定化: 極小面積をクランプ
        dual_areas_clamped = np.maximum(self.dual_areas, 1e-6)
        
        # 疎行列の要素を構築（対称版）
        # Δ̄_ij = -w_ij / sqrt(|v*_i| |v*_j|)
        for (vi, vj), weight in edge_weights.items():
            # 対称版の係数
            denominator = np.sqrt(dual_areas_clamped[vi] * dual_areas_clamped[vj])
            coeff = -weight / denominator
            
            # 非対角要素
            row_indices.append(vi)
            col_indices.append(vj)
            data_values.append(coeff)
            
            row_indices.append(vj)
            col_indices.append(vi)
            data_values.append(coeff)
        
        # 疎行列を構築（対角成分なし）
        Delta_bar_offdiag = csr_matrix((data_values, (row_indices, col_indices)), 
                                       shape=(self.n_vertices, self.n_vertices))
        
        # 対角要素: Δ̄_ii = -Σ_j Δ̄_ij（各行の和がゼロになるように）
        diagonal = -np.array(Delta_bar_offdiag.sum(axis=1)).flatten()
        
        # 対称化ラプラシアン Δ̄ を構築
        self.Delta_bar = Delta_bar_offdiag + diags(diagonal, format='csr')
        
        print(f"  完了")
        print(f"  行列サイズ: {self.Delta_bar.shape}")
        print(f"  非零要素数: {self.Delta_bar.nnz}")
        print(f"  スパース率: {self.Delta_bar.nnz / (self.n_vertices**2) * 100:.3f}%")
        
        return self.Delta_bar
    
    def _compute_nyquist_frequency(self):
        """
        Nyquist周波数を計算
        ω_nyquist = 1 / (2 × 平均エッジ長)
        
        Returns:
            float - Nyquist周波数
        """
        edge_lengths = []
        
        for face in self.faces:
            v0, v1, v2 = face
            p0 = self.vertices[v0]
            p1 = self.vertices[v1]
            p2 = self.vertices[v2]
            
            edge_lengths.append(np.linalg.norm(p1 - p0))
            edge_lengths.append(np.linalg.norm(p2 - p1))
            edge_lengths.append(np.linalg.norm(p0 - p2))
        
        mean_edge_length = np.mean(edge_lengths)
        omega_nyquist = 1.0 / (2.0 * mean_edge_length)
        
        return omega_nyquist
    
    def compute_spectrum(self, max_frequency=None, num_eigenpairs=None, 
                        batch_size=50, verbose=True):
        """
        Shift-Invert Lanczos法でスペクトル分解
        
        Algorithm (Vallet & Lévy 2008, Algorithm 1):
            λ_shift ← 0
            while (λ_last < ω²_max):
                Δ_shift = Δ̄ - λ_shift·I
                (eigenvectors, eigenvalues) = eigsh(Δ_shift⁻¹, k=batch_size, which='LM')
                λ = λ_shift + 1/eigenvalues (shift-invert変換)
                保存
                λ_shift を更新
        
        Parameters:
            max_frequency: float - 最大周波数（Noneなら自動計算）
            num_eigenpairs: int - 固有対の数（max_frequencyより優先）
            batch_size: int - 各イテレーションで計算する固有対数
            verbose: bool - 詳細表示
        """
        if self.Delta_bar is None:
            self.compute_laplacian()
        
        # Nyquist周波数を計算
        self.nyquist_frequency = self._compute_nyquist_frequency()
        
        # 最大周波数を決定
        if num_eigenpairs is not None:
            # 固有対の数が指定された場合はそれを優先
            target_num_eigenpairs = num_eigenpairs
            if verbose:
                print(f"\n固有対を{target_num_eigenpairs}個計算します")
        else:
            if max_frequency is None:
                max_frequency = self.nyquist_frequency
            
            if verbose:
                print(f"\nスペクトル分解を実行")
                print(f"  Nyquist周波数: {self.nyquist_frequency:.6f}")
                print(f"  目標最大周波数: {max_frequency:.6f}")
            
            # 必要な固有対数を推定（頂点数ベース）
            # 論文のTable 1に基づく経験則
            n = self.n_vertices
            if n < 10000:
                target_num_eigenpairs = min(100, n - 2)
            elif n < 50000:
                target_num_eigenpairs = min(300, n - 2)
            elif n < 100000:
                target_num_eigenpairs = min(500, n - 2)
            else:
                target_num_eigenpairs = min(800, n - 2)
            
            if verbose:
                print(f"  推定固有対数: {target_num_eigenpairs}")
        
        # Shift-Invert Lanczos法で反復的に計算
        lambda_shift = 0.0
        all_eigenvalues = []
        all_eigenvectors = []
        
        iteration = 0
        max_iterations = 50  # 安全装置
        
        while len(all_eigenvalues) < target_num_eigenpairs and iteration < max_iterations:
            iteration += 1
            
            if verbose:
                print(f"\n--- イテレーション {iteration} ---")
                print(f"  シフト値 λ_shift = {lambda_shift:.6e}")
            
            # シフトされた行列を構築
            Delta_shifted = self.Delta_bar - lambda_shift * eye(self.n_vertices)
            
            # 現在のバッチサイズを決定
            remaining = target_num_eigenpairs - len(all_eigenvalues)
            current_batch_size = min(batch_size, remaining, self.n_vertices - 2)
            
            if current_batch_size <= 0:
                break
            
            try:
                # Shift-Invert: Δ_shifted⁻¹ の最大固有値 = Δ̄ の最小固有値（シフト付近）
                # LU分解を使用
                lu_decomp = splu(Delta_shifted.tocsc())
                
                # 線形演算子を定義
                from scipy.sparse.linalg import LinearOperator
                
                def matvec(x):
                    return lu_decomp.solve(x)
                
                linear_op = LinearOperator(
                    shape=Delta_shifted.shape,
                    matvec=matvec,
                    dtype=Delta_shifted.dtype
                )
                
                # 固有値計算
                mu, vecs = eigsh(linear_op, k=current_batch_size, which='LM', 
                                maxiter=10000, tol=1e-6)
                
                # Shift-Invert変換を元に戻す: λ = λ_shift + 1/μ
                lambdas = lambda_shift + 1.0 / mu
                
                # 固有値で並び替え（昇順）
                sort_indices = np.argsort(lambdas)
                lambdas = lambdas[sort_indices]
                vecs = vecs[:, sort_indices]
                
                # 結果を保存
                for i in range(len(lambdas)):
                    # 既に計算済みの固有値と重複チェック
                    is_duplicate = False
                    for existing_lambda in all_eigenvalues:
                        if abs(lambdas[i] - existing_lambda) < 1e-8:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        all_eigenvalues.append(lambdas[i])
                        all_eigenvectors.append(vecs[:, i])
                
                if verbose:
                    print(f"  計算された固有値: {len(lambdas)}個")
                    print(f"  λ範囲: [{lambdas.min():.6e}, {lambdas.max():.6e}]")
                    print(f"  累積固有対数: {len(all_eigenvalues)}")
                
                # 次のシフト値を設定（現在の最大固有値より少し大きく）
                lambda_shift = lambdas.max() + 1e-6
                
            except Exception as e:
                if verbose:
                    print(f"  警告: イテレーション {iteration} で失敗: {e}")
                break
        
        # 固有値で並び替え
        sort_indices = np.argsort(all_eigenvalues)
        self.eigenvalues = np.array([all_eigenvalues[i] for i in sort_indices])
        self.eigenvectors = np.column_stack([all_eigenvectors[i] for i in sort_indices])
        
        # 周波数を計算
        self.frequencies = np.sqrt(np.maximum(self.eigenvalues, 0.0))
        
        # MHT（順変換）でフーリエ係数を計算
        self._compute_fourier_coefficients(verbose=verbose)
        
        if verbose:
            print(f"\n✅ スペクトル分解完了")
            print(f"  固有対数: {len(self.eigenvalues)}")
            print(f"  周波数範囲: [{self.frequencies.min():.6e}, {self.frequencies.max():.6e}]")
            print(f"  最大周波数/Nyquist: {self.frequencies.max()/self.nyquist_frequency:.2%}")
        
        return self.eigenvalues, self.eigenvectors
    
    def _compute_fourier_coefficients(self, verbose=True):
        """
        MHT（Manifold Harmonic Transform）でフーリエ係数を計算
        
        x̃_k = ⟨x, H_k⟩ = Σ_i x_i · H_k_i
        """
        if verbose:
            print("\nMHT（順変換）でフーリエ係数を計算中...")
        
        # 警告を抑制して内積を計算
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            self.fourier_X = self.eigenvectors.T @ self.vertices[:, 0]
            self.fourier_Y = self.eigenvectors.T @ self.vertices[:, 1]
            self.fourier_Z = self.eigenvectors.T @ self.vertices[:, 2]
        
        # nan/infをフィルタリング
        nan_mask_X = np.isnan(self.fourier_X) | np.isinf(self.fourier_X)
        nan_mask_Y = np.isnan(self.fourier_Y) | np.isinf(self.fourier_Y)
        nan_mask_Z = np.isnan(self.fourier_Z) | np.isinf(self.fourier_Z)
        
        if np.any(nan_mask_X):
            self.fourier_X[nan_mask_X] = 0.0
        if np.any(nan_mask_Y):
            self.fourier_Y[nan_mask_Y] = 0.0
        if np.any(nan_mask_Z):
            self.fourier_Z[nan_mask_Z] = 0.0
        
        if verbose:
            print(f"  完了")
            print(f"  フーリエ係数数: {len(self.fourier_X)}")
    
    def compute_residual(self, cutoff_harmonic=None, verbose=True):
        """
        高周波成分（残差）を計算
        
        residual = x_original - x_reconstructed
        
        Parameters:
            cutoff_harmonic: int - カットオフ調和関数番号（Noneなら全て）
            verbose: bool - 詳細表示
        
        Returns:
            np.ndarray - 残差 (n_vertices, 3)
        """
        if self.eigenvectors is None:
            raise ValueError("先に compute_spectrum() を実行してください")
        
        if cutoff_harmonic is None:
            cutoff_harmonic = len(self.fourier_X)
        else:
            cutoff_harmonic = min(cutoff_harmonic, len(self.fourier_X))
        
        if verbose:
            print(f"\n高周波成分を計算中（{cutoff_harmonic}個の調和関数まで使用）...")
        
        # 低周波成分で復元（逆MHT）
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            X_lowfreq = self.eigenvectors[:, :cutoff_harmonic] @ self.fourier_X[:cutoff_harmonic]
            Y_lowfreq = self.eigenvectors[:, :cutoff_harmonic] @ self.fourier_Y[:cutoff_harmonic]
            Z_lowfreq = self.eigenvectors[:, :cutoff_harmonic] @ self.fourier_Z[:cutoff_harmonic]
        
        # 残差 = 元の座標 - 低周波成分
        self.residual_X = self.vertices[:, 0] - X_lowfreq
        self.residual_Y = self.vertices[:, 1] - Y_lowfreq
        self.residual_Z = self.vertices[:, 2] - Z_lowfreq
        
        residual = np.column_stack([self.residual_X, self.residual_Y, self.residual_Z])
        
        if verbose:
            residual_magnitude = np.linalg.norm(residual, axis=1)
            print(f"  完了")
            print(f"  残差統計:")
            print(f"    平均振幅: {residual_magnitude.mean():.6e}")
            print(f"    最大振幅: {residual_magnitude.max():.6e}")
            print(f"    RMS:      {np.sqrt((residual_magnitude**2).mean()):.6e}")
        
        return residual
    
    def apply_lowpass_filter(self, cutoff_freq, include_residual=True, 
                            residual_weight=None, verbose=True):
        """
        ローパスフィルターを適用
        
        フィルター関数:
            F(ω) = 1  if ω ≤ ω_cutoff
                   0  if ω > ω_cutoff
        
        Parameters:
            cutoff_freq: float - カットオフ周波数
            include_residual: bool - 残差を含めるか
            residual_weight: float - 残差の重み（Noneなら0）
            verbose: bool - 詳細表示
        
        Returns:
            np.ndarray - フィルター適用後の頂点座標 (n_vertices, 3)
        """
        if self.eigenvectors is None:
            raise ValueError("先に compute_spectrum() を実行してください")
        
        if verbose:
            print(f"\nローパスフィルターを適用")
            print(f"  カットオフ周波数: {cutoff_freq:.6f}")
        
        # カットオフ周波数以下の調和関数を選択
        mask = self.frequencies <= cutoff_freq
        cutoff_harmonic = np.sum(mask)
        
        if cutoff_harmonic == 0:
            if verbose:
                print(f"  警告: カットオフ周波数以下の調和関数がありません")
            return self.vertices.copy()
        
        # フィルター適用（カットオフ以下は1、それ以上は0）
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            X_filtered = self.eigenvectors[:, :cutoff_harmonic] @ self.fourier_X[:cutoff_harmonic]
            Y_filtered = self.eigenvectors[:, :cutoff_harmonic] @ self.fourier_Y[:cutoff_harmonic]
            Z_filtered = self.eigenvectors[:, :cutoff_harmonic] @ self.fourier_Z[:cutoff_harmonic]
        
        filtered_vertices = np.column_stack([X_filtered, Y_filtered, Z_filtered])
        
        # 残差を追加
        if include_residual:
            if self.residual_X is None:
                self.compute_residual(cutoff_harmonic, verbose=False)
            
            if residual_weight is None:
                residual_weight = 0.0  # ローパスなので残差は含めない
            
            filtered_vertices[:, 0] += residual_weight * self.residual_X
            filtered_vertices[:, 1] += residual_weight * self.residual_Y
            filtered_vertices[:, 2] += residual_weight * self.residual_Z
        
        if verbose:
            print(f"  使用した調和関数数: {cutoff_harmonic}")
            print(f"  残差重み: {residual_weight}")
        
        return filtered_vertices
    
    def apply_highpass_filter(self, cutoff_freq, boost_factor=2.0, 
                             include_residual=True, verbose=True):
        """
        ハイパスフィルターを適用
        
        フィルター関数:
            F(ω) = 1               if ω ≤ ω_cutoff
                   boost_factor    if ω > ω_cutoff
        
        注: 低周波数を1に保つことでモデルの崩壊を防ぐ
        
        Parameters:
            cutoff_freq: float - カットオフ周波数
            boost_factor: float - 高周波数のブースト係数（1より大きい）
            include_residual: bool - 残差を含めるか
            verbose: bool - 詳細表示
        
        Returns:
            np.ndarray - フィルター適用後の頂点座標 (n_vertices, 3)
        """
        if self.eigenvectors is None:
            raise ValueError("先に compute_spectrum() を実行してください")
        
        if verbose:
            print(f"\nハイパスフィルターを適用")
            print(f"  カットオフ周波数: {cutoff_freq:.6f}")
            print(f"  高周波ブースト係数: {boost_factor:.2f}")
        
        # フィルター関数を各周波数に適用
        filter_weights = np.ones(len(self.frequencies))
        mask_high = self.frequencies > cutoff_freq
        filter_weights[mask_high] = boost_factor
        
        # 各成分にフィルターを適用
        filtered_coeffs_X = filter_weights * self.fourier_X
        filtered_coeffs_Y = filter_weights * self.fourier_Y
        filtered_coeffs_Z = filter_weights * self.fourier_Z
        
        # 逆MHT（ハイパスフィルター）
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            X_filtered = self.eigenvectors @ filtered_coeffs_X
            Y_filtered = self.eigenvectors @ filtered_coeffs_Y
            Z_filtered = self.eigenvectors @ filtered_coeffs_Z
        
        filtered_vertices = np.column_stack([X_filtered, Y_filtered, Z_filtered])
        
        # 残差を追加
        if include_residual:
            # カットオフ周波数での調和関数数を取得
            cutoff_harmonic = np.sum(self.frequencies <= cutoff_freq)
            
            if self.residual_X is None:
                self.compute_residual(cutoff_harmonic, verbose=False)
            
            # 残差の重みはブースト係数と同じ
            residual_weight = boost_factor
            
            filtered_vertices[:, 0] += residual_weight * self.residual_X
            filtered_vertices[:, 1] += residual_weight * self.residual_Y
            filtered_vertices[:, 2] += residual_weight * self.residual_Z
            
            if verbose:
                print(f"  残差重み: {residual_weight:.2f}")
        
        num_boosted = np.sum(mask_high)
        if verbose:
            print(f"  ブーストされた調和関数数: {num_boosted}/{len(self.frequencies)}")
        
        return filtered_vertices
    
    def apply_custom_filter(self, filter_func, include_residual=True, 
                           residual_weight=None, verbose=True):
        """
        カスタムフィルター関数を適用
        
        Parameters:
            filter_func: callable - フィルター関数 F(ω)
            include_residual: bool - 残差を含めるか
            residual_weight: float - 残差の重み（Noneなら自動計算）
            verbose: bool - 詳細表示
        
        Returns:
            np.ndarray - フィルター適用後の頂点座標 (n_vertices, 3)
        """
        if self.eigenvectors is None:
            raise ValueError("先に compute_spectrum() を実行してください")
        
        if verbose:
            print(f"\nカスタムフィルターを適用")
        
        # 各周波数にフィルターを適用
        filter_weights = np.array([filter_func(omega) for omega in self.frequencies])
        
        # 各成分にフィルターを適用
        filtered_coeffs_X = filter_weights * self.fourier_X
        filtered_coeffs_Y = filter_weights * self.fourier_Y
        filtered_coeffs_Z = filter_weights * self.fourier_Z
        
        # 逆MHT（カスタムフィルター）
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            X_filtered = self.eigenvectors @ filtered_coeffs_X
            Y_filtered = self.eigenvectors @ filtered_coeffs_Y
            Z_filtered = self.eigenvectors @ filtered_coeffs_Z
        
        filtered_vertices = np.column_stack([X_filtered, Y_filtered, Z_filtered])
        
        # 残差を追加
        if include_residual:
            if self.residual_X is None:
                # デフォルトは全調和関数で計算
                self.compute_residual(len(self.fourier_X), verbose=False)
            
            if residual_weight is None:
                # 高周波域（Nyquist付近）でのフィルター平均値を計算
                omega_max = self.frequencies[-1]
                omega_samples = np.linspace(omega_max * 0.8, self.nyquist_frequency, 100)
                filter_samples = np.array([filter_func(w) for w in omega_samples])
                residual_weight = np.mean(filter_samples)
            
            filtered_vertices[:, 0] += residual_weight * self.residual_X
            filtered_vertices[:, 1] += residual_weight * self.residual_Y
            filtered_vertices[:, 2] += residual_weight * self.residual_Z
            
            if verbose:
                print(f"  残差重み: {residual_weight:.4f}")
        
        if verbose:
            print(f"  フィルター範囲: [{filter_weights.min():.4f}, {filter_weights.max():.4f}]")
        
        return filtered_vertices


# ユーティリティ関数

def estimate_memory_requirement(n_vertices, num_eigenpairs):
    """
    必要なメモリ量を推定（MB単位）
    
    Parameters:
        n_vertices: int - 頂点数
        num_eigenpairs: int - 固有対の数
    
    Returns:
        dict - メモリ推定値
    """
    mem = {}
    
    # 頂点データ (n_vertices, 3) float64
    mem['vertices'] = (n_vertices * 3 * 8) / (1024**2)
    
    # 固有ベクトル (n_vertices, num_eigenpairs) float64
    mem['eigenvectors'] = (n_vertices * num_eigenpairs * 8) / (1024**2)
    
    # 固有値 (num_eigenpairs,) float64
    mem['eigenvalues'] = (num_eigenpairs * 8) / (1024**2)
    
    # フーリエ係数 3つ (num_eigenpairs,) float64
    mem['fourier_coeffs'] = (num_eigenpairs * 3 * 8) / (1024**2)
    
    # 残差 (n_vertices, 3) float64
    mem['residual'] = (n_vertices * 3 * 8) / (1024**2)
    
    # 疎行列（推定: 各頂点あたり平均7個のエッジ）
    avg_edges_per_vertex = 7
    nnz = n_vertices * avg_edges_per_vertex
    mem['laplacian_sparse'] = (nnz * (8 + 4 + 4)) / (1024**2)  # data + indices + indptr
    
    mem['total'] = sum(mem.values())
    
    return mem
