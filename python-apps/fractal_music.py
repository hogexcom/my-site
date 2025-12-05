"""
Fractal Music Composition using Fourier Filtering (Web Version)

フーリエフィルタリングによるフラクタル音楽の作曲:
1. パワースペクトル密度 P(ω) ∝ ω^(-beta) に従うランダム信号を生成
2. 逆フーリエ変換で時間領域の波形を得る
3. 音高（ピッチ）や音量のシーケンスとして解釈
4. MIDIバイト列として出力（Web用）

理論的背景:
- 1/f ノイズ（ピンクノイズ）: beta ≈ 1
- ブラウンノイズ: beta ≈ 2
- ホワイトノイズ: beta ≈ 0
- フラクタル音楽では beta = 2H + 1 (H: Hurst指数)

音楽への応用:
- H = 0.5 (beta = 2): 自然なランダムウォーク的メロディ
- H > 0.5: 持続性のある滑らかなメロディ
- H < 0.5: 反復性のあるギザギザしたメロディ
"""

import numpy as np
import struct
from io import BytesIO


class FractalMusicComposer:
    """フーリエフィルタリングによるフラクタル音楽生成（Web用）"""
    
    def __init__(self, hurst_index=0.5, seed=None):
        """
        Parameters:
            hurst_index: float - Hurst指数（0 < H < 1）
            seed: int - 乱数シード
        """
        self.hurst_index = hurst_index
        self.beta = 2 * hurst_index + 1
        
        if seed is not None:
            np.random.seed(seed)
    
    def generate_fractal_signal(self, num_samples):
        """
        フラクタルな信号を生成
        
        パワースペクトル密度: P(ω) ∝ ω^(-beta)
        
        Parameters:
            num_samples: int - サンプル数
        
        Returns:
            signal: np.ndarray - 生成された信号 (実数)
        """
        N = num_samples
        # 奇数個のサンプルとする（対称性のため）
        if N % 2 == 0:
            N += 1
        
        # フーリエ係数を生成
        A_0 = np.zeros(1, dtype=complex)
        A_pos = np.zeros((N // 2,), dtype=complex)
        
        for i in range(N // 2):
            omega = i + 1  # 周波数インデックス（0を避ける）
            
            # パワースペクトル密度 P(ω) ∝ ω^(-beta) にするため
            # 振幅は ω^(-beta/2) とする
            amplitude = np.power(omega, -self.beta / 2) * np.random.normal(0, 1)
            
            # ランダムな位相
            phase = 2 * np.pi * np.random.random()
            
            A_pos[i] = amplitude * np.exp(1j * phase)
        
        # 負の周波数成分は正の周波数の複素共役（実信号を保証）
        A_neg = np.conjugate(A_pos[::-1])
        
        # フーリエ係数を結合
        fourier_coeffs = np.concatenate((A_0, A_pos, A_neg))
        
        # 逆フーリエ変換
        signal = np.fft.ifft(fourier_coeffs).real
        
        return signal[:num_samples]
    
    def signal_to_notes(self, signal, num_notes, pitch_range=(48, 84)):
        """
        信号を音符のピッチに変換
        
        Parameters:
            signal: np.ndarray - 入力信号
            num_notes: int - 音符数
            pitch_range: tuple - (最小MIDI音高, 最大MIDI音高)
        
        Returns:
            pitches: np.ndarray - MIDI音高の配列
        """
        min_pitch, max_pitch = pitch_range
        
        # リサンプリング
        indices = np.linspace(0, len(signal) - 1, num_notes)
        signal_resampled = np.interp(indices, np.arange(len(signal)), signal)
        
        # 信号を [0, 1] に正規化
        signal_norm = (signal_resampled - signal_resampled.min()) / (signal_resampled.max() - signal_resampled.min() + 1e-10)
        
        # MIDI音高範囲にマッピング
        pitches = min_pitch + signal_norm * (max_pitch - min_pitch)
        pitches = np.round(pitches).astype(int)
        
        return pitches
    
    def signal_to_velocities(self, signal, num_notes, min_velocity=40, max_velocity=100, scale=1.0):
        """
        信号を音量（ベロシティ）に変換
        
        Parameters:
            signal: np.ndarray - 入力信号
            num_notes: int - 音符数
            min_velocity: int - 最小ベロシティ
            max_velocity: int - 最大ベロシティ
            scale: float - スケール（0.0-1.0）
        
        Returns:
            velocities: np.ndarray - ベロシティの配列
        """
        # リサンプリング
        indices = np.linspace(0, len(signal) - 1, num_notes)
        signal_resampled = np.interp(indices, np.arange(len(signal)), signal)
        
        # 信号の絶対値を使用
        signal_abs = np.abs(signal_resampled)
        
        # [min_velocity, max_velocity] にマッピング
        signal_norm = (signal_abs - signal_abs.min()) / (signal_abs.max() - signal_abs.min() + 1e-10)
        velocities = min_velocity + signal_norm * (max_velocity - min_velocity) * scale
        velocities = np.clip(velocities, min_velocity, max_velocity).astype(int)
        
        return velocities
    
    def create_midi(self, pitches, velocities, note_duration=0.5, tempo=120, instrument=0):
        """
        MIDIバイト列を作成（シングルトラック）
        
        Parameters:
            pitches: np.ndarray - MIDI音高の配列
            velocities: np.ndarray - ベロシティの配列
            note_duration: float - 音符の長さ（秒）
            tempo: int - テンポ（BPM）
            instrument: int - MIDI楽器番号
        
        Returns:
            bytes - MIDIファイルのバイト列
        """
        return self._create_midi_bytes(
            [(pitches, velocities, instrument)],
            note_duration=note_duration,
            tempo=tempo
        )
    
    def create_midi_multi_track(self, pitches_list, velocities_list, instruments,
                                note_duration=0.5, tempo=120):
        """
        マルチトラックMIDIバイト列を作成
        
        Parameters:
            pitches_list: list of np.ndarray - 各トラックのMIDI音高
            velocities_list: list of np.ndarray - 各トラックのベロシティ
            instruments: list of int - 各トラックの楽器番号
            note_duration: float - 音符の長さ（秒）
            tempo: int - テンポ（BPM）
        
        Returns:
            bytes - MIDIファイルのバイト列
        """
        tracks = []
        for pitches, velocities, instrument in zip(pitches_list, velocities_list, instruments):
            tracks.append((pitches, velocities, instrument))
        
        return self._create_midi_bytes(tracks, note_duration=note_duration, tempo=tempo)
    
    def create_midi_multi_track_varied(self, pitches_list, velocities_list, instruments,
                                       durations_list, tempo=120):
        """
        トラックごとに異なる音符長を持つマルチトラックMIDIバイト列を作成
        
        Parameters:
            pitches_list: list of np.ndarray - 各トラックのMIDI音高
            velocities_list: list of np.ndarray - 各トラックのベロシティ
            instruments: list of int - 各トラックの楽器番号
            durations_list: list of float - 各トラックの音符の長さ（秒）
            tempo: int - テンポ（BPM）
        
        Returns:
            bytes - MIDIファイルのバイト列
        """
        tracks = []
        for pitches, velocities, instrument, duration in zip(
            pitches_list, velocities_list, instruments, durations_list
        ):
            tracks.append((pitches, velocities, instrument, duration))
        
        return self._create_midi_bytes_varied(tracks, tempo=tempo)
    
    def _create_midi_bytes(self, tracks, note_duration=0.5, tempo=120):
        """
        MIDIバイト列を直接生成
        
        Parameters:
            tracks: list of (pitches, velocities, instrument) tuples
            note_duration: float - 音符の長さ（秒）
            tempo: int - テンポ（BPM）
        
        Returns:
            bytes - MIDIファイルのバイト列
        """
        # PPQ (Pulses Per Quarter note)
        ppq = 480
        
        # 音符の長さをティック数に変換
        ticks_per_note = int(ppq * note_duration * tempo / 60)
        
        output = BytesIO()
        
        # MIDIヘッダー
        # フォーマット1（複数トラック）
        num_tracks = len(tracks) + 1  # トラック + テンポトラック
        header = b'MThd'
        header += struct.pack('>I', 6)  # チャンクサイズ
        header += struct.pack('>H', 1)  # フォーマット1
        header += struct.pack('>H', num_tracks)  # トラック数
        header += struct.pack('>H', ppq)  # PPQ
        output.write(header)
        
        # テンポトラック
        tempo_track = BytesIO()
        # テンポメタイベント（マイクロ秒/四分音符）
        microseconds_per_beat = 60000000 // tempo
        tempo_track.write(self._variable_length(0))  # デルタタイム
        tempo_track.write(b'\xFF\x51\x03')  # テンポメタイベント
        tempo_track.write(struct.pack('>I', microseconds_per_beat)[1:])  # 3バイト
        # End of Track
        tempo_track.write(self._variable_length(0))
        tempo_track.write(b'\xFF\x2F\x00')
        
        tempo_data = tempo_track.getvalue()
        output.write(b'MTrk')
        output.write(struct.pack('>I', len(tempo_data)))
        output.write(tempo_data)
        
        # 各トラック
        for track_idx, (pitches, velocities, instrument) in enumerate(tracks):
            track_data = BytesIO()
            channel = track_idx % 16  # チャンネル（0-15）
            
            # プログラムチェンジ（楽器選択）
            track_data.write(self._variable_length(0))
            track_data.write(bytes([0xC0 | channel, instrument]))
            
            # 音符を追加
            for i, (pitch, velocity) in enumerate(zip(pitches, velocities)):
                # Note On
                delta = 0 if i == 0 else 0  # Note Offの後すぐにNote On
                track_data.write(self._variable_length(delta))
                track_data.write(bytes([0x90 | channel, int(pitch), int(velocity)]))
                
                # Note Off
                track_data.write(self._variable_length(ticks_per_note))
                track_data.write(bytes([0x80 | channel, int(pitch), 0]))
            
            # End of Track
            track_data.write(self._variable_length(0))
            track_data.write(b'\xFF\x2F\x00')
            
            track_bytes = track_data.getvalue()
            output.write(b'MTrk')
            output.write(struct.pack('>I', len(track_bytes)))
            output.write(track_bytes)
        
        return output.getvalue()
    
    def _create_midi_bytes_varied(self, tracks, tempo=120):
        """
        トラックごとに異なる音符長でMIDIバイト列を生成
        
        Parameters:
            tracks: list of (pitches, velocities, instrument, duration) tuples
            tempo: int - テンポ（BPM）
        
        Returns:
            bytes - MIDIファイルのバイト列
        """
        # PPQ (Pulses Per Quarter note)
        ppq = 480
        
        output = BytesIO()
        
        # MIDIヘッダー
        num_tracks = len(tracks) + 1  # トラック + テンポトラック
        header = b'MThd'
        header += struct.pack('>I', 6)
        header += struct.pack('>H', 1)  # フォーマット1
        header += struct.pack('>H', num_tracks)
        header += struct.pack('>H', ppq)
        output.write(header)
        
        # テンポトラック
        tempo_track = BytesIO()
        microseconds_per_beat = 60000000 // tempo
        tempo_track.write(self._variable_length(0))
        tempo_track.write(b'\xFF\x51\x03')
        tempo_track.write(struct.pack('>I', microseconds_per_beat)[1:])
        tempo_track.write(self._variable_length(0))
        tempo_track.write(b'\xFF\x2F\x00')
        
        tempo_data = tempo_track.getvalue()
        output.write(b'MTrk')
        output.write(struct.pack('>I', len(tempo_data)))
        output.write(tempo_data)
        
        # 各トラック（それぞれ異なる音符長）
        for track_idx, (pitches, velocities, instrument, note_duration) in enumerate(tracks):
            track_data = BytesIO()
            channel = track_idx % 16
            
            # 音符の長さをティック数に変換
            ticks_per_note = int(ppq * note_duration * tempo / 60)
            
            # プログラムチェンジ
            track_data.write(self._variable_length(0))
            track_data.write(bytes([0xC0 | channel, instrument]))
            
            # 音符を追加
            for i, (pitch, velocity) in enumerate(zip(pitches, velocities)):
                # Note On
                delta = 0 if i == 0 else 0
                track_data.write(self._variable_length(delta))
                track_data.write(bytes([0x90 | channel, int(pitch), int(velocity)]))
                
                # Note Off
                track_data.write(self._variable_length(ticks_per_note))
                track_data.write(bytes([0x80 | channel, int(pitch), 0]))
            
            # End of Track
            track_data.write(self._variable_length(0))
            track_data.write(b'\xFF\x2F\x00')
            
            track_bytes = track_data.getvalue()
            output.write(b'MTrk')
            output.write(struct.pack('>I', len(track_bytes)))
            output.write(track_bytes)
        
        return output.getvalue()
    
    def _variable_length(self, value):
        """可変長数値をエンコード"""
        if value == 0:
            return bytes([0])
        
        result = []
        while value > 0:
            byte = value & 0x7F
            value >>= 7
            result.append(byte)
        
        result.reverse()
        for i in range(len(result) - 1):
            result[i] |= 0x80
        
        return bytes(result)


# テスト用
if __name__ == '__main__':
    composer = FractalMusicComposer(hurst_index=0.7)
    signal = composer.generate_fractal_signal(400)
    notes = composer.signal_to_notes(signal, 100, pitch_range=(48, 84))
    velocities = composer.signal_to_velocities(signal, 100)
    
    midi_data = composer.create_midi(notes, velocities, note_duration=0.5)
    print(f"Generated MIDI: {len(midi_data)} bytes")
    
    # マルチトラック
    midi_multi = composer.create_midi_multi_track(
        [notes, notes],
        [velocities, velocities],
        [0, 32],  # Piano, Bass
        note_duration=0.5
    )
    print(f"Generated Multi-track MIDI: {len(midi_multi)} bytes")
