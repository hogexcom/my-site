"""
Spectral Mesh Processing テストプログラム

spectral_mesh_processing.py のテストと使用例
"""

import numpy as np
import sys
import os

from spectral_mesh_processing import SpectralMeshProcessor, estimate_memory_requirement
from Laplace_de_Rham_operator_on_0_forms import TriangleMesh


def test_bunny_model():
    """
    バニーモデルでスペクトル処理をテスト
    """
    print("=" * 80)
    print("Spectral Mesh Processing テスト")
    print("=" * 80)
    
    # OBJファイルを読み込み
    obj_file = "../3d_models/bunny.obj"
    if not os.path.exists(obj_file):
        print(f"エラー: {obj_file} が見つかりません")
        return
    
    print(f"\n📁 メッシュを読み込み中: {obj_file}")
    mesh = TriangleMesh.from_obj(obj_file)
    
    print(f"  頂点数: {mesh.n_vertices}")
    print(f"  面数: {mesh.n_faces}")
    
    # メモリ推定
    num_eigenpairs = 300
    mem_estimate = estimate_memory_requirement(mesh.n_vertices, num_eigenpairs)
    print(f"\n💾 推定メモリ使用量 ({num_eigenpairs}固有対):")
    for key, value in mem_estimate.items():
        print(f"  {key}: {value:.2f} MB")
    
    # スペクトル処理クラスを初期化
    print("\n🔧 SpectralMeshProcessor を初期化")
    processor = SpectralMeshProcessor(
        vertices=mesh.vertices,
        faces=mesh.faces,
        dual_type='circumcentric'
    )
    
    # スペクトル分解を実行
    print("\n📊 スペクトル分解を実行")
    eigenvalues, eigenvectors = processor.compute_spectrum(
        num_eigenpairs=num_eigenpairs,
        batch_size=50,
        verbose=True
    )
    
    # 実際のメモリ使用量を確認
    print("\n💾 実際のメモリ使用量:")
    actual_usage = processor.get_memory_usage()
    for key, value in actual_usage.items():
        print(f"  {key}: {value:.2f} MB")
    
    # 残差を計算
    print("\n🔍 残差を計算")
    residual = processor.compute_residual(
        cutoff_harmonic=200,
        verbose=True
    )
    
    # ローパスフィルターを適用
    print("\n🔽 ローパスフィルターを適用")
    cutoff_freq = processor.nyquist_frequency * 0.3
    lowpass_vertices = processor.apply_lowpass_filter(
        cutoff_freq=cutoff_freq,
        include_residual=False,
        verbose=True
    )
    
    # ローパス結果を保存
    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "bunny_lowpass.obj")
    save_obj(output_file, lowpass_vertices, mesh.faces)
    print(f"  💾 保存: {output_file}")
    
    # ハイパスフィルターを適用
    print("\n🔼 ハイパスフィルターを適用")
    highpass_vertices = processor.apply_highpass_filter(
        cutoff_freq=cutoff_freq,
        boost_factor=2.5,
        include_residual=True,
        verbose=True
    )
    
    # ハイパス結果を保存
    output_file = os.path.join(output_dir, "bunny_highpass.obj")
    save_obj(output_file, highpass_vertices, mesh.faces)
    print(f"  💾 保存: {output_file}")
    
    # カスタムフィルター: バンドパス
    print("\n🎚️  カスタムフィルター（バンドパス）を適用")
    
    def bandpass_filter(omega):
        """バンドパスフィルター: 中間周波数のみ通過"""
        omega_low = processor.nyquist_frequency * 0.2
        omega_high = processor.nyquist_frequency * 0.6
        
        if omega_low <= omega <= omega_high:
            return 2.0  # 中間周波数を強調
        else:
            return 0.5  # その他は減衰
    
    bandpass_vertices = processor.apply_custom_filter(
        filter_func=bandpass_filter,
        include_residual=False,
        verbose=True
    )
    
    # バンドパス結果を保存
    output_file = os.path.join(output_dir, "bunny_bandpass.obj")
    save_obj(output_file, bandpass_vertices, mesh.faces)
    print(f"  💾 保存: {output_file}")
    
    # 統計情報を表示
    print("\n📈 フィルター結果の統計")
    print_mesh_stats("元のメッシュ", mesh.vertices)
    print_mesh_stats("ローパス", lowpass_vertices)
    print_mesh_stats("ハイパス", highpass_vertices)
    print_mesh_stats("バンドパス", bandpass_vertices)
    
    print("\n" + "=" * 80)
    print("✅ テスト完了")
    print("=" * 80)


def save_obj(filename, vertices, faces):
    """
    OBJファイルに保存
    
    Parameters:
        filename: str - 出力ファイル名
        vertices: np.ndarray (n, 3) - 頂点座標
        faces: np.ndarray (m, 3) - 面の頂点インデックス（0-indexed）
    """
    with open(filename, 'w') as f:
        # 頂点を書き込み
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # 面を書き込み（1-indexedに変換）
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def print_mesh_stats(name, vertices):
    """メッシュの統計情報を表示"""
    print(f"\n  {name}:")
    print(f"    X範囲: [{vertices[:, 0].min():.4f}, {vertices[:, 0].max():.4f}]")
    print(f"    Y範囲: [{vertices[:, 1].min():.4f}, {vertices[:, 1].max():.4f}]")
    print(f"    Z範囲: [{vertices[:, 2].min():.4f}, {vertices[:, 2].max():.4f}]")
    center = vertices.mean(axis=0)
    print(f"    中心: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    
    # エッジの長さ統計（サンプリング）
    from_center = vertices - center
    distances = np.linalg.norm(from_center, axis=1)
    print(f"    中心からの距離: [{distances.min():.4f}, {distances.max():.4f}] (平均: {distances.mean():.4f})")


def test_simple_mesh():
    """
    シンプルなメッシュ（立方体）でテスト
    """
    print("=" * 80)
    print("シンプルメッシュ（立方体）テスト")
    print("=" * 80)
    
    # 立方体の頂点
    vertices = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
    ], dtype=np.float64)
    
    # 立方体の面（三角形化）
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # 底面
        [4, 6, 5], [4, 7, 6],  # 上面
        [0, 4, 5], [0, 5, 1],  # 前面
        [2, 6, 7], [2, 7, 3],  # 背面
        [0, 3, 7], [0, 7, 4],  # 左面
        [1, 5, 6], [1, 6, 2],  # 右面
    ], dtype=np.int32)
    
    print(f"\n頂点数: {len(vertices)}")
    print(f"面数: {len(faces)}")
    
    # スペクトル処理
    processor = SpectralMeshProcessor(vertices, faces)
    
    # メモリ推定
    mem = estimate_memory_requirement(len(vertices), num_eigenpairs=6)
    print(f"\n推定メモリ: {mem['total']:.4f} MB")
    
    # スペクトル分解（小さいので全固有対を計算）
    eigenvalues, eigenvectors = processor.compute_spectrum(
        num_eigenpairs=6,  # 8頂点 → 最大8個の固有対
        verbose=True
    )
    
    print(f"\n固有値:")
    for i, ev in enumerate(eigenvalues):
        print(f"  λ_{i} = {ev:.6e} (ω = {np.sqrt(ev):.6e})")
    
    # ローパスフィルター
    cutoff_freq = processor.frequencies[3]  # 4つ目の周波数でカット
    lowpass = processor.apply_lowpass_filter(cutoff_freq, verbose=True)
    
    # 結果を保存
    os.makedirs("out", exist_ok=True)
    save_obj("out/cube_original.obj", vertices, faces)
    save_obj("out/cube_lowpass.obj", lowpass, faces)
    
    print("\n✅ シンプルメッシュテスト完了")


def compare_filters():
    """
    複数のフィルターを比較
    """
    print("=" * 80)
    print("フィルター比較テスト")
    print("=" * 80)
    
    # メッシュ読み込み
    obj_file = "../3d_models/bunny.obj"
    if not os.path.exists(obj_file):
        print(f"エラー: {obj_file} が見つかりません")
        return
    
    mesh = TriangleMesh.from_obj(obj_file)
    processor = SpectralMeshProcessor(mesh.vertices, mesh.faces)
    
    # スペクトル分解
    print("\nスペクトル分解中...")
    processor.compute_spectrum(num_eigenpairs=200, verbose=False)
    
    # 複数のカットオフ周波数でローパスフィルターをテスト
    cutoff_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print(f"\nNyquist周波数: {processor.nyquist_frequency:.6f}")
    print("\nローパスフィルター比較:")
    
    os.makedirs("out", exist_ok=True)
    
    for ratio in cutoff_ratios:
        cutoff_freq = processor.nyquist_frequency * ratio
        filtered = processor.apply_lowpass_filter(cutoff_freq, verbose=False)
        
        # 元のメッシュとの差分を計算
        diff = filtered - mesh.vertices
        diff_magnitude = np.linalg.norm(diff, axis=1)
        
        print(f"\n  カットオフ比率: {ratio:.1f} (ω = {cutoff_freq:.6f})")
        print(f"    使用調和関数: {np.sum(processor.frequencies <= cutoff_freq)}/{len(processor.frequencies)}")
        print(f"    平均変位: {diff_magnitude.mean():.6e}")
        print(f"    最大変位: {diff_magnitude.max():.6e}")
        
        # 保存
        output_file = f"out/bunny_lowpass_{int(ratio*100)}.obj"
        save_obj(output_file, filtered, mesh.faces)
        print(f"    保存: {output_file}")
    
    # ハイパスフィルター比較
    print("\n\nハイパスフィルター比較:")
    boost_factors = [1.5, 2.0, 3.0, 5.0]
    cutoff_freq = processor.nyquist_frequency * 0.3
    
    for boost in boost_factors:
        filtered = processor.apply_highpass_filter(
            cutoff_freq=cutoff_freq,
            boost_factor=boost,
            verbose=False
        )
        
        diff = filtered - mesh.vertices
        diff_magnitude = np.linalg.norm(diff, axis=1)
        
        print(f"\n  ブースト係数: {boost:.1f}")
        print(f"    平均変位: {diff_magnitude.mean():.6e}")
        print(f"    最大変位: {diff_magnitude.max():.6e}")
        
        output_file = f"out/bunny_highpass_boost{int(boost*10)}.obj"
        save_obj(output_file, filtered, mesh.faces)
        print(f"    保存: {output_file}")
    
    print("\n✅ フィルター比較完了")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Spectral Mesh Processing テスト")
    parser.add_argument('--test', choices=['bunny', 'cube', 'compare', 'all'], 
                       default='bunny',
                       help='実行するテスト (default: bunny)')
    
    args = parser.parse_args()
    
    if args.test == 'bunny':
        test_bunny_model()
    elif args.test == 'cube':
        test_simple_mesh()
    elif args.test == 'compare':
        compare_filters()
    elif args.test == 'all':
        test_simple_mesh()
        print("\n\n")
        test_bunny_model()
        print("\n\n")
        compare_filters()
