import { useState, useRef, useCallback, useEffect } from 'react'
import type { MutableRefObject } from 'react'
import type { Track } from './App'
import * as mm from '@magenta/music'
import * as Tone from 'tone'

interface MidiPlayerProps {
  tracks: Track[]
  onDownload: () => void
  onStopRef?: MutableRefObject<(() => void) | null>
}

// Magenta SoundFontPlayerを使用したMIDIプレイヤー
class SoundFontMidiPlayer {
  private player: mm.SoundFontPlayer | null = null

  constructor() {
    // SoundFontPlayerの作成（GoogleのSoundFontを使用）
    this.player = new mm.SoundFontPlayer(
      'https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus'
    )
  }

  async playTracks(tracks: Track[]): Promise<void> {
    if (!this.player) {
      throw new Error('SoundFont Player not initialized')
    }

    this.stop()

    // AudioContextを開始（ユーザーアクション後に呼ばれる必要がある）
    await Tone.start()
    console.log('Tone.start() completed')

    // 全トラックのノートを結合（各ノートにprogram情報を付与）
    const allNotes: mm.NoteSequence.INote[] = []
    
    tracks.forEach(track => {
      track.notes.forEach(note => {
        allNotes.push({
          pitch: note.pitch,
          startTime: note.time,
          endTime: note.time + note.duration,
          velocity: note.velocity,
          program: track.instrument,
          isDrum: false
        })
      })
    })

    // NoteSequenceに変換
    const noteSequence: mm.INoteSequence = {
      notes: allNotes,
      totalTime: Math.max(...allNotes.map(n => n.endTime || 0))
    }

    // 必要なサンプルだけをロードして再生
    await this.player.loadSamples(noteSequence)
    console.log('Samples loaded, starting playback')
    // start()は再生開始後にPromiseを返すが、再生終了を待たない
    try {
      this.player.start(noteSequence)
      console.log('start() called successfully')
    } catch (e) {
      console.error('start() error:', e)
      throw e
    }
  }

  stop(): void {
    if (this.player) {
      this.player.stop()
    }
  }

  getTotalDuration(tracks: Track[]): number {
    let maxEnd = 0
    tracks.forEach(track => {
      track.notes.forEach(note => {
        const end = note.time + note.duration
        if (end > maxEnd) maxEnd = end
      })
    })
    return maxEnd
  }

  dispose(): void {
    this.stop()
    this.player = null
  }
}

// MIDI楽器名のマッピング
const INSTRUMENT_NAMES: Record<number, string> = {
  0: 'アコースティックグランドピアノ',
  1: 'ブライトアコースティックピアノ',
  4: 'エレクトリックピアノ1',
  5: 'エレクトリックピアノ2',
  6: 'ハープシコード',
  7: 'クラビネット',
  11: 'ビブラフォン',
  13: 'マリンバ',
  16: 'ドローバーオルガン',
  17: 'パーカッシブオルガン',
  18: 'ロックオルガン',
  19: 'チャーチオルガン',
  24: 'アコースティックギター（ナイロン）',
  25: 'アコースティックギター（スチール）',
  26: 'ジャズギター',
  27: 'クリーンギター',
  28: 'ミュートギター',
  29: 'オーバードライブギター',
  32: 'アコースティックベース',
  33: 'エレクトリックベース（フィンガー）',
  34: 'エレクトリックベース（ピック）',
  35: 'フレットレスベース',
  36: 'スラップベース1',
  37: 'スラップベース2',
  38: 'シンセベース1',
  39: 'シンセベース2',
  40: 'バイオリン',
  41: 'ビオラ',
  42: 'チェロ',
  43: 'コントラバス',
  46: 'オーケストラハープ',
  48: '弦楽アンサンブル1',
  49: '弦楽アンサンブル2',
  50: 'シンセストリングス1',
  51: 'シンセストリングス2',
  52: '声のああ',
  53: '声のおお',
  54: 'シンセボイス',
  56: 'トランペット',
  57: 'トロンボーン',
  58: 'チューバ',
  60: 'フレンチホルン',
  65: 'アルトサックス',
  66: 'テナーサックス',
  68: 'オーボエ',
  71: 'クラリネット',
  73: 'フルート',
  74: 'リコーダー',
  75: 'パンフルート',
  79: 'オカリナ',
  80: 'リードシンセ（矩形波）',
  81: 'リードシンセ（ノコギリ波）',
  87: 'リードシンセ（ベースリード）',
  88: 'パッドシンセ（ニューエイジ）',
  89: 'パッドシンセ（ウォーム）',
  90: 'パッドシンセ（ポリシンセ）',
  91: 'パッドシンセ（コワイヤ）',
  92: 'パッドシンセ（ボウド）',
  93: 'パッドシンセ（メタリック）',
  94: 'パッドシンセ（ハロー）',
  95: 'パッドシンセ（スイープ）',
}

function MidiPlayer({ tracks, onDownload, onStopRef }: MidiPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const playerRef = useRef<SoundFontMidiPlayer | null>(null)
  const playbackTimerRef = useRef<number | null>(null)

  const handleStop = useCallback(() => {
    if (!playerRef.current) return

    playerRef.current.stop()
    setIsPlaying(false)

    if (playbackTimerRef.current) {
      clearTimeout(playbackTimerRef.current)
      playbackTimerRef.current = null
    }
  }, [])

  useEffect(() => {
    playerRef.current = new SoundFontMidiPlayer()
    return () => {
      playerRef.current?.dispose()
    }
  }, [])

  // onStopRef に handleStop を登録
  useEffect(() => {
    if (onStopRef) {
      onStopRef.current = handleStop
    }
    return () => {
      if (onStopRef) {
        onStopRef.current = null
      }
    }
  }, [onStopRef, handleStop])

  const handlePlay = useCallback(async () => {
    if (!playerRef.current) {
      console.error('Player not initialized')
      return
    }

    const player = playerRef.current
    
    setIsLoading(true)
    console.log('Starting playback, loading samples...')
    
    try {
      await player.playTracks(tracks)
      console.log('Playback started')
      setIsLoading(false)
      setIsPlaying(true)

      // 再生終了タイマー
      const duration = player.getTotalDuration(tracks) * 1000
      console.log(`Setting end timer for ${duration}ms`)
      playbackTimerRef.current = window.setTimeout(() => {
        setIsPlaying(false)
        console.log('Playback ended')
      }, duration + 500)
    } catch (error) {
      console.error('Playback error:', error)
      setIsLoading(false)
      setIsPlaying(false)
    }
  }, [tracks])

  const formatTrackInfo = (track: Track): string => {
    const instrumentName = INSTRUMENT_NAMES[track.instrument] || `楽器${track.instrument}`
    return `${track.name} (${instrumentName}): ${track.notes.length}音符`
  }

  return (
    <div className="player-container">
      <h3>🎹 再生（SoundFont音源）</h3>
      
      <div className="track-info">
        {tracks.map((track, index) => (
          <div key={index} className="track-item">
            {formatTrackInfo(track)}
          </div>
        ))}
      </div>

      <div className="buttons">
        {isLoading ? (
          <button className="btn btn-secondary" disabled>
            🔄 音源ロード中...
          </button>
        ) : !isPlaying ? (
          <button className="btn btn-success" onClick={handlePlay}>
            ▶ 再生
          </button>
        ) : (
          <button className="btn btn-danger" onClick={handleStop}>
            ⏹ 停止
          </button>
        )}
        
        <button className="btn btn-secondary" onClick={onDownload}>
          💾 MIDIダウンロード
        </button>
      </div>
    </div>
  )
}

export default MidiPlayer
