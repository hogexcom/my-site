interface ControlPanelProps {
  hurstIndex: number
  onHurstIndexChange: (value: number) => void
  songDuration: number
  onSongDurationChange: (value: number) => void
  numTracks: number
  onNumTracksChange: (value: number) => void
  noteDuration: number
  onNoteDurationChange: (value: number) => void
  pitchRangeMin: number
  onPitchRangeMinChange: (value: number) => void
  pitchRangeMax: number
  onPitchRangeMaxChange: (value: number) => void
  melodyInstrument: number
  onMelodyInstrumentChange: (value: number) => void
  bassInstrument: number
  onBassInstrumentChange: (value: number) => void
  chordInstrument: number
  onChordInstrumentChange: (value: number) => void
  onGenerate: () => void
  onRandomGenerate: () => void
  isGenerating: boolean
  isReady: boolean
}

// 楽器リスト（MIDI Program Number）- エクスポートして外部から使えるように
export const MELODY_INSTRUMENTS = [
  { value: 0, label: 'アコースティックグランドピアノ' },
  { value: 1, label: 'ブライトアコースティックピアノ' },
  { value: 4, label: 'エレクトリックピアノ1' },
  { value: 5, label: 'エレクトリックピアノ2' },
  { value: 6, label: 'ハープシコード' },
  { value: 11, label: 'ビブラフォン' },
  { value: 13, label: 'マリンバ' },
  { value: 24, label: 'アコースティックギター（ナイロン）' },
  { value: 25, label: 'アコースティックギター（スチール）' },
  { value: 40, label: 'バイオリン' },
  { value: 41, label: 'ビオラ' },
  { value: 42, label: 'チェロ' },
  { value: 46, label: 'オーケストラハープ' },
  { value: 56, label: 'トランペット' },
  { value: 57, label: 'トロンボーン' },
  { value: 60, label: 'フレンチホルン' },
  { value: 65, label: 'アルトサックス' },
  { value: 66, label: 'テナーサックス' },
  { value: 68, label: 'オーボエ' },
  { value: 71, label: 'クラリネット' },
  { value: 73, label: 'フルート' },
  { value: 74, label: 'リコーダー' },
  { value: 75, label: 'パンフルート' },
  { value: 79, label: 'オカリナ' },
  { value: 80, label: 'リードシンセ（矩形波）' },
  { value: 81, label: 'リードシンセ（ノコギリ波）' },
]

export const BASS_INSTRUMENTS = [
  { value: 32, label: 'アコースティックベース' },
  { value: 33, label: 'エレキベース（フィンガー）' },
  { value: 34, label: 'エレキベース（ピック）' },
  { value: 35, label: 'フレットレスベース' },
  { value: 36, label: 'スラップベース1' },
  { value: 37, label: 'スラップベース2' },
  { value: 38, label: 'シンセベース1' },
  { value: 39, label: 'シンセベース2' },
  { value: 42, label: 'チェロ' },
  { value: 43, label: 'コントラバス' },
  { value: 58, label: 'チューバ' },
  { value: 87, label: 'リードシンセ（ベースリード）' },
]

export const CHORD_INSTRUMENTS = [
  { value: 0, label: 'アコースティックグランドピアノ' },
  { value: 1, label: 'ブライトアコースティックピアノ' },
  { value: 4, label: 'エレクトリックピアノ1' },
  { value: 5, label: 'エレクトリックピアノ2' },
  { value: 6, label: 'ハープシコード' },
  { value: 7, label: 'クラビネット' },
  { value: 16, label: 'ドローバーオルガン' },
  { value: 17, label: 'パーカッシブオルガン' },
  { value: 18, label: 'ロックオルガン' },
  { value: 19, label: 'チャーチオルガン' },
  { value: 24, label: 'アコースティックギター（ナイロン）' },
  { value: 25, label: 'アコースティックギター（スチール）' },
  { value: 26, label: 'ジャズギター' },
  { value: 27, label: 'クリーンギター' },
  { value: 28, label: 'ミュートギター' },
  { value: 29, label: 'オーバードライブギター' },
  { value: 48, label: '弦楽アンサンブル1' },
  { value: 49, label: '弦楽アンサンブル2' },
  { value: 50, label: 'シンセストリングス1' },
  { value: 51, label: 'シンセストリングス2' },
  { value: 52, label: '声のああ' },
  { value: 53, label: '声のおお' },
  { value: 54, label: 'シンセボイス' },
  { value: 88, label: 'パッドシンセ（ニューエイジ）' },
  { value: 89, label: 'パッドシンセ（ウォーム）' },
  { value: 90, label: 'パッドシンセ（ポリシンセ）' },
  { value: 91, label: 'パッドシンセ（コワイヤ）' },
  { value: 92, label: 'パッドシンセ（ボウド）' },
  { value: 93, label: 'パッドシンセ（メタリック）' },
  { value: 94, label: 'パッドシンセ（ハロー）' },
  { value: 95, label: 'パッドシンセ（スイープ）' },
]

function ControlPanel({
  hurstIndex,
  onHurstIndexChange,
  songDuration,
  onSongDurationChange,
  numTracks,
  onNumTracksChange,
  noteDuration,
  onNoteDurationChange,
  pitchRangeMin,
  onPitchRangeMinChange,
  pitchRangeMax,
  onPitchRangeMaxChange,
  melodyInstrument,
  onMelodyInstrumentChange,
  bassInstrument,
  onBassInstrumentChange,
  chordInstrument,
  onChordInstrumentChange,
  onGenerate,
  onRandomGenerate,
  isGenerating,
  isReady
}: ControlPanelProps) {
  return (
    <div className="controls">
      <div className="control-group">
        <label>
          Hurst指数 (H): <span>{hurstIndex.toFixed(2)}</span>
          <div className="help-text">
            H &lt; 0.5: 反持続性（変動が激しい）
            <br />
            H = 0.5: ランダムウォーク
            <br />
            H &gt; 0.5: 持続性（滑らか）
          </div>
        </label>
        <input
          type="range"
          min="0.01"
          max="0.99"
          step="0.01"
          value={hurstIndex}
          onChange={(e) => onHurstIndexChange(parseFloat(e.target.value))}
        />
      </div>

      <div className="control-group">
        <label>
          曲の長さ: <span>{songDuration}秒</span>
        </label>
        <input
          type="range"
          min="10"
          max="120"
          step="5"
          value={songDuration}
          onChange={(e) => onSongDurationChange(parseInt(e.target.value))}
        />
      </div>

      <div className="control-group">
        <label>
          トラック数: <span>{numTracks}</span>
        </label>
        <select
          value={numTracks}
          onChange={(e) => onNumTracksChange(parseInt(e.target.value))}
        >
          <option value={1}>1 (メロディのみ)</option>
          <option value={2}>2 (メロディ + ベース)</option>
          <option value={3}>3 (メロディ + ベース + コード)</option>
        </select>
      </div>

      <div className="control-group">
        <label>
          メロディの音符の長さ: <span>{noteDuration.toFixed(2)}秒</span>
          <div className="help-text">
            ベースは4倍、コードは2倍の長さになります
          </div>
        </label>
        <input
          type="range"
          min="0.1"
          max="1.0"
          step="0.05"
          value={noteDuration}
          onChange={(e) => onNoteDurationChange(parseFloat(e.target.value))}
        />
      </div>

      <div className="control-group">
        <label>
          音域 (MIDI): <span>{pitchRangeMin} - {pitchRangeMax}</span>
          <div className="help-text">
            48 = C3, 60 = C4 (中央のド), 72 = C5, 84 = C6
          </div>
        </label>
        <div className="range-inputs">
          <input
            type="range"
            min="24"
            max="84"
            step="1"
            value={pitchRangeMin}
            onChange={(e) => {
              const val = parseInt(e.target.value)
              if (val < pitchRangeMax) onPitchRangeMinChange(val)
            }}
          />
          <input
            type="range"
            min="36"
            max="108"
            step="1"
            value={pitchRangeMax}
            onChange={(e) => {
              const val = parseInt(e.target.value)
              if (val > pitchRangeMin) onPitchRangeMaxChange(val)
            }}
          />
        </div>
      </div>

      <div className="control-group">
        <label>メロディ楽器</label>
        <select
          value={melodyInstrument}
          onChange={(e) => onMelodyInstrumentChange(parseInt(e.target.value))}
        >
          {MELODY_INSTRUMENTS.map(inst => (
            <option key={inst.value} value={inst.value}>{inst.label}</option>
          ))}
        </select>
      </div>

      {numTracks >= 2 && (
        <div className="control-group">
          <label>ベース楽器</label>
          <select
            value={bassInstrument}
            onChange={(e) => onBassInstrumentChange(parseInt(e.target.value))}
          >
            {BASS_INSTRUMENTS.map(inst => (
              <option key={inst.value} value={inst.value}>{inst.label}</option>
            ))}
          </select>
        </div>
      )}

      {numTracks >= 3 && (
        <div className="control-group">
          <label>コード楽器</label>
          <select
            value={chordInstrument}
            onChange={(e) => onChordInstrumentChange(parseInt(e.target.value))}
          >
            {CHORD_INSTRUMENTS.map(inst => (
              <option key={inst.value} value={inst.value}>{inst.label}</option>
            ))}
          </select>
        </div>
      )}

      <div className="buttons">
        <button
          className="btn btn-primary"
          onClick={onGenerate}
          disabled={!isReady || isGenerating}
        >
          {!isReady ? 'Loading Python...' : isGenerating ? '生成中...' : '🎵 生成'}
        </button>
        <button
          className="btn btn-secondary"
          onClick={onRandomGenerate}
          disabled={!isReady || isGenerating}
        >
          {isGenerating ? '生成中...' : '🎲 ランダム生成'}
        </button>
      </div>
    </div>
  )
}

export default ControlPanel
