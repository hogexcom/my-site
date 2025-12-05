import { useState, useCallback, useRef } from 'react'
import { usePyodide } from '../../hooks/usePyodide'
import ControlPanel, { MELODY_INSTRUMENTS, BASS_INSTRUMENTS, CHORD_INSTRUMENTS } from './ControlPanel'
import SignalChart from './SignalChart'
import MidiPlayer from './MidiPlayer'
import './App.css'

export interface Note {
  pitch: number
  time: number
  duration: number
  velocity: number
}

export interface Track {
  name: string
  instrument: number
  notes: Note[]
}

export interface GeneratedMusic {
  signal: number[]
  trackSignals: number[][]  // 各トラックの信号
  tracks: Track[]
  midiBase64: string
}

function App() {
  const [hurstIndex, setHurstIndex] = useState(0.5)  // デフォルト0.5（ランダムウォーク）
  const [songDuration, setSongDuration] = useState(30)  // 曲の長さ（秒）
  const [numTracks, setNumTracks] = useState(3)  // デフォルト3トラック
  const [noteDuration, setNoteDuration] = useState(0.25)  // メロディの1音の長さ（秒）
  const [pitchRangeMin, setPitchRangeMin] = useState(48)
  const [pitchRangeMax, setPitchRangeMax] = useState(84)
  const [melodyInstrument, setMelodyInstrument] = useState(73)  // フルート
  const [bassInstrument, setBassInstrument] = useState(33)  // エレキベース
  const [chordInstrument, setChordInstrument] = useState(4)  // エレクトリックピアノ
  
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedMusic, setGeneratedMusic] = useState<GeneratedMusic | null>(null)
  const stopPlaybackRef = useRef<(() => void) | null>(null)
  
  const { pyodide, isReady, error: pyodideError } = usePyodide([
    'fractal_music.py'
  ])

  const handleGenerate = useCallback(async () => {
    if (!pyodide || !isReady) return
    
    // 再生中なら停止
    if (stopPlaybackRef.current) {
      stopPlaybackRef.current()
    }
    
    setIsGenerating(true)
    
    try {
      await pyodide.runPythonAsync(`
import numpy as np
import base64
from fractal_music import FractalMusicComposer

# パラメータ設定
hurst_index = ${hurstIndex}
song_duration = ${songDuration}  # 曲の長さ（秒）
num_tracks = ${numTracks}
melody_note_duration = ${noteDuration}  # メロディの1音の長さ（秒）
pitch_range = (${pitchRangeMin}, ${pitchRangeMax})

composer = FractalMusicComposer(hurst_index=hurst_index)

# 各トラックの信号を保存
track_signals = []

if num_tracks == 1:
    # シングルトラック - メロディ用（ピアノ）
    num_notes = int(song_duration / melody_note_duration)
    signal = composer.generate_fractal_signal(num_notes * 4)
    track_signals.append(signal[:num_notes].tolist())
    notes = composer.signal_to_notes(signal, num_notes, pitch_range=pitch_range)
    velocities = composer.signal_to_velocities(signal, num_notes)
    
    track_data = [{
        'name': 'Melody',
        'instrument': 0,  # アコースティックグランドピアノ
        'notes': [
            {
                'pitch': int(notes[i]),
                'time': float(i * melody_note_duration),
                'duration': float(melody_note_duration),
                'velocity': int(velocities[i])
            }
            for i in range(len(notes))
        ]
    }]
    
    midi_data = composer.create_midi(notes, velocities, note_duration=melody_note_duration)
else:
    # マルチトラック - 音楽的に適切な楽器と音符長を選択
    # Melody: 短め（指定値）、Chords: やや長め（3倍）、Bass: 長め（6倍）
    melody_instrument = ${melodyInstrument}
    bass_instrument = ${bassInstrument}
    chord_instrument = ${chordInstrument}
    
    track_configs = [
        {'name': 'Melody', 'pitch_range': pitch_range, 'velocity_scale': 1.0, 'instrument': melody_instrument, 'duration_mult': 1.0},
        {'name': 'Bass', 'pitch_range': (28, 48), 'velocity_scale': 0.9, 'instrument': bass_instrument, 'duration_mult': 6.0},
    ]
    if num_tracks >= 3:
        track_configs.append({'name': 'Chords', 'pitch_range': (48, 72), 'velocity_scale': 0.7, 'instrument': chord_instrument, 'duration_mult': 3.0})
    
    track_data = []
    pitches_list = []
    velocities_list = []
    durations_list = []
    
    for i, config in enumerate(track_configs):
        # 各トラック用の信号（曲の長さから音符数を計算）
        track_note_duration = melody_note_duration * config['duration_mult']
        track_num_notes = max(1, int(song_duration / track_note_duration))
        
        track_signal = composer.generate_fractal_signal(track_num_notes * 4)
        track_signals.append(track_signal[:track_num_notes].tolist())
        notes = composer.signal_to_notes(track_signal, track_num_notes, pitch_range=config['pitch_range'])
        vels = composer.signal_to_velocities(track_signal, track_num_notes, scale=config['velocity_scale'])
        
        pitches_list.append(notes)
        velocities_list.append(vels)
        durations_list.append(track_note_duration)
        
        track_data.append({
            'name': config['name'],
            'instrument': config['instrument'],
            'notes': [
                {
                    'pitch': int(notes[j]),
                    'time': float(j * track_note_duration),
                    'duration': float(track_note_duration * 0.9),  # 少し短めにしてスタッカート感
                    'velocity': int(vels[j])
                }
                for j in range(len(notes))
            ]
        })
    
    midi_data = composer.create_midi_multi_track_varied(
        pitches_list,
        velocities_list,
        [config['instrument'] for config in track_configs],
        durations_list
    )
    signal = track_signals[0]  # 最初のトラックの信号をメインとして使用

# 結果を出力用に変換
result_signal = signal if isinstance(signal, list) else signal.tolist()
result_track_signals = track_signals
result_tracks = track_data
result_midi_base64 = base64.b64encode(midi_data).decode('utf-8')

print(f"Generated {len(result_tracks)} track(s), song duration: {song_duration}s")
      `)
      
      const signal = pyodide.globals.get('result_signal').toJs()
      const trackSignals = pyodide.globals.get('result_track_signals').toJs()
      const tracks = pyodide.globals.get('result_tracks').toJs()
      const midiBase64 = pyodide.globals.get('result_midi_base64')
      
      // Convert tracks from Map to plain objects with proper typing
      const tracksArray: Track[] = Array.from(tracks).map((track: any) => {
        const trackObj = Object.fromEntries(track)
        const notes: Note[] = Array.from(trackObj.notes).map((note: any) => {
          const noteObj = Object.fromEntries(note)
          return {
            pitch: noteObj.pitch as number,
            time: noteObj.time as number,
            duration: noteObj.duration as number,
            velocity: noteObj.velocity as number
          }
        })
        return {
          name: trackObj.name as string,
          instrument: trackObj.instrument as number,
          notes
        }
      })
      
      // Convert trackSignals
      const trackSignalsArray: number[][] = Array.from(trackSignals).map((sig: any) => 
        Array.from(sig) as number[]
      )
      
      setGeneratedMusic({
        signal: Array.from(signal) as number[],
        trackSignals: trackSignalsArray,
        tracks: tracksArray,
        midiBase64
      })
    } catch (error) {
      console.error('Generation error:', error)
    } finally {
      setIsGenerating(false)
    }
  }, [pyodide, isReady, hurstIndex, songDuration, numTracks, noteDuration, pitchRangeMin, pitchRangeMax, melodyInstrument, bassInstrument, chordInstrument])

  const handleDownloadMidi = useCallback(() => {
    if (!generatedMusic?.midiBase64) return
    
    const binaryString = atob(generatedMusic.midiBase64)
    const bytes = new Uint8Array(binaryString.length)
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i)
    }
    
    const blob = new Blob([bytes], { type: 'audio/midi' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `fractal_music_H${hurstIndex.toFixed(2)}.mid`
    a.click()
    URL.revokeObjectURL(url)
  }, [generatedMusic, hurstIndex])

  // ランダム生成: 楽器をランダムに変更してから生成
  const handleRandomGenerate = useCallback(() => {
    // ランダムに楽器を選択
    const randomMelody = MELODY_INSTRUMENTS[Math.floor(Math.random() * MELODY_INSTRUMENTS.length)].value
    const randomBass = BASS_INSTRUMENTS[Math.floor(Math.random() * BASS_INSTRUMENTS.length)].value
    const randomChord = CHORD_INSTRUMENTS[Math.floor(Math.random() * CHORD_INSTRUMENTS.length)].value
    
    setMelodyInstrument(randomMelody)
    setBassInstrument(randomBass)
    setChordInstrument(randomChord)
    
    // stateの更新後に生成を実行するためにタイムアウトを使用
    setTimeout(() => {
      handleGenerate()
    }, 0)
  }, [handleGenerate])

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🎵 Fractal Music Generator</h1>
        <p className="subtitle">フラクタル理論に基づく自動作曲</p>
      </header>

      {pyodideError && (
        <div className="error-message">
          Error: {pyodideError}
        </div>
      )}

      <ControlPanel
        hurstIndex={hurstIndex}
        onHurstIndexChange={setHurstIndex}
        songDuration={songDuration}
        onSongDurationChange={setSongDuration}
        numTracks={numTracks}
        onNumTracksChange={setNumTracks}
        noteDuration={noteDuration}
        onNoteDurationChange={setNoteDuration}
        pitchRangeMin={pitchRangeMin}
        onPitchRangeMinChange={setPitchRangeMin}
        pitchRangeMax={pitchRangeMax}
        onPitchRangeMaxChange={setPitchRangeMax}
        melodyInstrument={melodyInstrument}
        onMelodyInstrumentChange={setMelodyInstrument}
        bassInstrument={bassInstrument}
        onBassInstrumentChange={setBassInstrument}
        chordInstrument={chordInstrument}
        onChordInstrumentChange={setChordInstrument}
        onGenerate={handleGenerate}
        onRandomGenerate={handleRandomGenerate}
        isGenerating={isGenerating}
        isReady={isReady}
      />

      {generatedMusic && (
        <>
          <SignalChart 
            trackSignals={generatedMusic.trackSignals}
            trackNames={generatedMusic.tracks.map(t => t.name)}
            hurstIndex={hurstIndex}
          />
          
          <MidiPlayer
            tracks={generatedMusic.tracks}
            onDownload={handleDownloadMidi}
            onStopRef={stopPlaybackRef}
          />
        </>
      )}

      <footer className="app-footer">
        <p>
          <a href={import.meta.env.BASE_URL}>← Back to Apps</a>
        </p>
      </footer>
    </div>
  )
}

export default App
