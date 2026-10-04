import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { apiGet, DanceDetail, posterUrl, ScoreEvent, videoUrl, wsUrl } from '../api'
import PoseGlyph from './PoseGlyph'

type MovePreview = {
  t_ms: number
  landmarks: Array<{x: number; y: number; z?: number; v?: number}>
}

type PlaybackInfo = {
  duration_ms: number
  mirror_mode: boolean
  tempo: number
  beat_ms: number[]
  strong_beat_ms: number[]
  events: Array<{type: string; start_ms: number; end_ms: number}>
  theme?: Record<string, any>
  move_previews?: MovePreview[]
}

function fmt(ms: number) {
  const total = Math.max(0, Math.floor(ms / 1000))
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

function gradeClass(g: string) { return g ? `grade-${g.toLowerCase()}` : '' }

export default function Gameplay() {
  const { danceId, sessionId } = useParams<{ danceId: string; sessionId: string }>()
  const navigate = useNavigate()
  const [score, setScore] = useState(0)
  const [combo, setCombo] = useState(0)
  const [grade, setGrade] = useState('')
  const [timingOffset, setTimingOffset] = useState<number | null>(null)
  const [holdState, setHoldState] = useState<string | null>(null)
  const [trackingLost, setTrackingLost] = useState(false)
  const [phoneConnected, setPhoneConnected] = useState(true)
  const [progress, setProgress] = useState(0)
  const [mediaMs, setMediaMs] = useState(0)
  const [gameOver, setGameOver] = useState(false)
  const [results, setResults] = useState<any>(null)
  const [ready, setReady] = useState(false)
  const [countdown, setCountdown] = useState<string | null>('READY')
  const [paused, setPaused] = useState(false)
  const [started, setStarted] = useState(false)
  const [detail, setDetail] = useState<DanceDetail | null>(null)
  const [playback, setPlayback] = useState<PlaybackInfo | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const rafRef = useRef<number | null>(null)
  const lastClockSentRef = useRef(0)
  const gradeTimeoutRef = useRef<any>(null)

  const handleScoreEvent = useCallback((event: ScoreEvent) => {
    setScore(prev => event.total_score ?? (prev + event.score))
    setCombo(event.combo)
    setGrade(event.grade)
    setTimingOffset(event.timing_offset_ms ?? null)
    setHoldState(event.hold_state || null)
    setTrackingLost(Boolean(event.tracking_lost))
    if (gradeTimeoutRef.current) clearTimeout(gradeTimeoutRef.current)
    gradeTimeoutRef.current = setTimeout(() => setGrade(''), 720)
  }, [])

  useEffect(() => {
    if (!danceId) return
    Promise.all([
      apiGet(`/api/dances/${danceId}`),
      apiGet(`/api/dances/${danceId}/playback`),
    ]).then(([dance, play]) => {
      setDetail(dance)
      setPlayback(play)
    }).catch(console.error)
  }, [danceId])

  useEffect(() => {
    if (!sessionId) return
    const ws = new WebSocket(wsUrl(`/ws/game/${sessionId}`))
    wsRef.current = ws
    ws.onopen = () => setReady(true)
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data)
      if (msg.type === 'score_event') handleScoreEvent(msg)
      else if (msg.type === 'phone_connected') setPhoneConnected(true)
      else if (msg.type === 'phone_disconnected') { setPhoneConnected(false); setTrackingLost(true) }
      else if (msg.type === 'game_over') { setGameOver(true); setResults(msg.results) }
    }
    ws.onclose = () => setReady(false)
    return () => {
      ws.close()
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      if (gradeTimeoutRef.current) clearTimeout(gradeTimeoutRef.current)
    }
  }, [sessionId, handleScoreEvent])

  const clockLoop = useCallback((now: number) => {
    const video = videoRef.current
    const ws = wsRef.current
    if (!video || !ws || ws.readyState !== WebSocket.OPEN || video.paused || video.ended) {
      rafRef.current = requestAnimationFrame(clockLoop)
      return
    }
    const current = Math.max(0, Math.round(video.currentTime * 1000))
    setMediaMs(current)
    setProgress(Math.min(100, current / Math.max(1, playback?.duration_ms || detail?.duration_ms || 1) * 100))
    if (now - lastClockSentRef.current >= 45) {
      ws.send(JSON.stringify({ action: 'media_clock', media_time_ms: current }))
      lastClockSentRef.current = now
    }
    rafRef.current = requestAnimationFrame(clockLoop)
  }, [playback, detail])

  const startGame = async () => {
    if (!videoRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
    for (const value of ['3', '2', '1']) {
      setCountdown(value)
      await new Promise(resolve => setTimeout(resolve, 650))
    }
    setCountdown('GO!')
    await new Promise(resolve => setTimeout(resolve, 340))
    setCountdown(null)
    videoRef.current.currentTime = 0
    videoRef.current.muted = false
    await videoRef.current.play()
    wsRef.current.send(JSON.stringify({ action: 'start', media_time_ms: 0 }))
    setStarted(true)
    lastClockSentRef.current = 0
    rafRef.current = requestAnimationFrame(clockLoop)
  }

  const togglePause = async () => {
    const video = videoRef.current
    const ws = wsRef.current
    if (!video || !ws || ws.readyState !== WebSocket.OPEN) return
    if (video.paused) {
      await video.play(); ws.send(JSON.stringify({ action: 'resume' })); setPaused(false)
    } else {
      video.pause(); ws.send(JSON.stringify({ action: 'pause' })); setPaused(true)
    }
  }

  const stopGame = () => {
    videoRef.current?.pause()
    wsRef.current?.send(JSON.stringify({ action: 'stop' }))
  }

  const durationMs = playback?.duration_ms || detail?.duration_ms || 1
  const timingText = timingOffset == null ? '' : Math.abs(timingOffset) < 45 ? 'ON BEAT' : Math.abs(timingOffset) < 90 ? 'CLOSE' : `${Math.abs(timingOffset)}ms`
  const stars = Math.max(0, Math.min(5, Math.floor(score / 30000)))

  const nextMoves = useMemo(() => {
    const moves = playback?.move_previews || []
    if (!moves.length) return [] as MovePreview[]
    let idx = moves.findIndex(m => m.t_ms >= mediaMs + 260)
    if (idx < 0) idx = Math.max(0, moves.length - 3)
    return moves.slice(idx, idx + 3)
  }, [playback?.move_previews, mediaMs])

  const timelineMarkers = useMemo(() => {
    const source = playback?.strong_beat_ms?.length ? playback.strong_beat_ms : playback?.beat_ms || []
    if (!source.length) return [12, 31, 50, 69, 87]
    const count = Math.min(7, source.length)
    return Array.from({length: count}, (_, i) => {
      const idx = Math.min(source.length - 1, Math.round(i * (source.length - 1) / Math.max(1, count - 1)))
      return Math.max(3, Math.min(97, source[idx] / durationMs * 100))
    })
  }, [playback, durationMs])

  if (gameOver && results) {
    const hits = Object.values(results.grade_counts || {}).reduce((a: number, b: any) => a + Number(b || 0), 0) as number
    const quality = hits ? ((results.grade_counts?.perfect || 0) + .85 * (results.grade_counts?.super || 0) + .65 * (results.grade_counts?.good || 0) + .4 * (results.grade_counts?.ok || 0)) / hits : 0
    const resultStars = Math.max(1, Math.min(5, Math.round(quality * 5)))
    return (
      <main className="ref-results">
        <div className="ref-results-card">
          <div className="ref-brand compact">DANCE<span>FLOW</span></div>
          <p>ROUTINE COMPLETE</p>
          <h1>{Number(results.total_score || 0).toLocaleString()}</h1>
          <div className="ref-results-stars">{'★'.repeat(resultStars)}<span>{'★'.repeat(5 - resultStars)}</span></div>
          <div className="ref-result-grid">
            <div><b>{results.max_combo || 0}x</b><span>MAX COMBO</span></div>
            <div><b>{Math.round((results.accuracy_arms || 0) * 100)}%</b><span>ARMS</span></div>
            <div><b>{Math.round((results.accuracy_torso || 0) * 100)}%</b><span>TORSO</span></div>
            <div><b>{Math.round((results.accuracy_legs || 0) * 100)}%</b><span>LEGS</span></div>
          </div>
          <div className="ref-results-actions"><button className="ref-primary" onClick={() => navigate(`/connect/${danceId}`)}>▶ Dance again</button><button className="ref-secondary" onClick={() => navigate('/')}>Library</button></div>
        </div>
      </main>
    )
  }

  return (
    <main className="ref-gameplay">
      {danceId && (
        <video
          ref={videoRef}
          src={videoUrl(danceId)}
          className="ref-game-video"
          style={{ transform: detail?.mirror_mode ? 'scaleX(-1)' : undefined }}
          playsInline
          preload="auto"
          onEnded={() => wsRef.current?.send(JSON.stringify({ action: 'media_clock', media_time_ms: durationMs }))}
        />
      )}
      <div className="ref-game-vignette" />

      <section className="ref-song-panel">
        <div className="ref-song-cover">{danceId && detail?.has_poster ? <img src={posterUrl(danceId)} /> : <span>DF</span>}</div>
        <div className="ref-song-copy">
          <h1>{detail?.title || 'Loading routine…'}</h1>
          <p>DANCEFLOW CREW <i>│</i> POP <i>│</i> PLAYABLE</p>
          <div className="ref-song-progress"><b style={{width:`${progress}%`}} /><em /></div>
        </div>
        <div className="ref-song-time">{fmt(mediaMs)} / {fmt(durationMs)}</div>
      </section>

      <section className="ref-score-panel">
        <div className="ref-score-copy"><span>♕ SCORE</span><strong>{score.toLocaleString()}</strong></div>
        <div className="ref-stars" aria-label={`${stars} stars`}>
          {[0,1,2,3,4].map(i => <b key={i} className={i < stars ? 'earned' : ''}>★</b>)}
        </div>
        <div className="ref-combo"><span>COMBO</span><strong>{combo}</strong></div>
      </section>

      <aside className="ref-next-moves">
        {[0,1,2].map((i) => (
          <div key={i} className={`ref-move-card ${i === 0 ? 'active' : ''}`}>
            <PoseGlyph pose={nextMoves[i]?.landmarks} active={i === 0} variant={i} />
          </div>
        ))}
        <span>NEXT<br/>MOVES</span>
      </aside>

      {grade && !trackingLost && (
        <div className={`ref-grade ${gradeClass(grade)}`}>
          <span>♕</span><strong>{grade === 'x' ? 'X' : grade.toUpperCase()}</strong><em>{timingText || 'ON BEAT'}</em>
        </div>
      )}
      {holdState && ['entering','holding'].includes(holdState) && <div className="ref-hold">HOLD!</div>}
      {holdState === 'yeah' && <div className="ref-hold">YEAH!</div>}

      <section className="ref-timeline">
        <div className="ref-wave ref-wave-left">{Array.from({length:13},(_,i)=><i key={i} style={{height:`${10 + ((i*13)%30)}px`}} />)}</div>
        <div className="ref-timeline-track">
          <div className="ref-timeline-fill" style={{width:`${progress}%`}} />
          {timelineMarkers.map((p,i) => <i key={i} className="ref-timeline-marker" style={{left:`${p}%`}} />)}
          <b className="ref-timeline-cursor" style={{left:`${progress}%`}} />
        </div>
        <div className="ref-wave ref-wave-right">{Array.from({length:13},(_,i)=><i key={i} style={{height:`${10 + (((12-i)*13)%30)}px`}} />)}</div>
      </section>

      {trackingLost && <div className="ref-tracking-warning"><b>STEP INTO FRAME</b><span>Tracking paused — no points are being awarded.</span></div>}
      {!phoneConnected && <div className="ref-phone-warning">PHONE DISCONNECTED</div>}

      {countdown && (
        <div className="ref-start-overlay">
          <div className="ref-start-card">
            <div className="ref-brand compact">DANCE<span>FLOW</span></div>
            <strong className="ref-countdown">{countdown}</strong>
            {countdown === 'READY' && <>
              <p>Phone tracking ready. Keep your full body in frame.</p>
              <button disabled={!ready || !detail || !playback} onClick={startGame} className="ref-primary">▶ START DANCE</button>
            </>}
          </div>
        </div>
      )}

      {started && !gameOver && (
        <div className="ref-game-controls"><button onClick={togglePause}>{paused ? '▶' : 'Ⅱ'}</button><button onClick={stopGame}>■</button></div>
      )}
    </main>
  )
}
