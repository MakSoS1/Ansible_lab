import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { apiGet, DanceDetail, ScoreEvent, videoUrl, wsUrl } from '../api'

const CONNECTIONS: Array<[number, number]> = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28], [27, 29], [29, 31], [28, 30], [30, 32],
]

type PlaybackInfo = {
  duration_ms: number
  mirror_mode: boolean
  tempo: number
  beat_ms: number[]
  strong_beat_ms: number[]
  events: Array<{type: string; start_ms: number; end_ms: number}>
}

export default function Gameplay() {
  const { danceId, sessionId } = useParams<{ danceId: string; sessionId: string }>()
  const navigate = useNavigate()
  const [score, setScore] = useState(0)
  const [combo, setCombo] = useState(0)
  const [grade, setGrade] = useState('')
  const [similarity, setSimilarity] = useState(0)
  const [timingOffset, setTimingOffset] = useState<number | null>(null)
  const [holdState, setHoldState] = useState<string | null>(null)
  const [trackingLost, setTrackingLost] = useState(false)
  const [phoneConnected, setPhoneConnected] = useState(true)
  const [progress, setProgress] = useState(0)
  const [gameOver, setGameOver] = useState(false)
  const [results, setResults] = useState<any>(null)
  const [ready, setReady] = useState(false)
  const [countdown, setCountdown] = useState<string | null>('READY')
  const [paused, setPaused] = useState(false)
  const [started, setStarted] = useState(false)
  const [beatPulse, setBeatPulse] = useState(0)
  const [detail, setDetail] = useState<DanceDetail | null>(null)
  const [playback, setPlayback] = useState<PlaybackInfo | null>(null)
  const [limbScores, setLimbScores] = useState<Record<string, number>>({})

  const wsRef = useRef<WebSocket | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number | null>(null)
  const lastClockSentRef = useRef(0)
  const gradeTimeoutRef = useRef<any>(null)
  const beatIndexRef = useRef(0)

  const beats = useMemo(() => playback?.strong_beat_ms?.length ? playback.strong_beat_ms : playback?.beat_ms || [], [playback])

  const drawCoachPose = useCallback((pose?: Array<{x: number; y: number; v?: number}>) => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return
    const rect = video.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    const w = Math.max(1, Math.floor(rect.width * dpr))
    const h = Math.max(1, Math.floor(rect.height * dpr))
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w
      canvas.height = h
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, w, h)
    if (!pose || pose.length < 33) return

    ctx.save()
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.shadowBlur = 28 * dpr
    ctx.shadowColor = 'rgba(0, 238, 255, .9)'
    ctx.strokeStyle = 'rgba(117, 249, 255, .90)'
    ctx.lineWidth = 8 * dpr
    CONNECTIONS.forEach(([a, b]) => {
      const pa = pose[a], pb = pose[b]
      if (!pa || !pb || (pa.v ?? 1) < .25 || (pb.v ?? 1) < .25) return
      ctx.beginPath()
      ctx.moveTo(pa.x * w, pa.y * h)
      ctx.lineTo(pb.x * w, pb.y * h)
      ctx.stroke()
    })
    ctx.shadowBlur = 18 * dpr
    ctx.fillStyle = 'rgba(255, 255, 255, .95)'
    ;[11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28].forEach(i => {
      const p = pose[i]
      if (!p || (p.v ?? 1) < .25) return
      ctx.beginPath()
      ctx.arc(p.x * w, p.y * h, 5 * dpr, 0, Math.PI * 2)
      ctx.fill()
    })
    ctx.restore()
  }, [])

  const handleScoreEvent = useCallback((event: ScoreEvent) => {
    setScore(prev => event.total_score ?? (prev + event.score))
    setCombo(event.combo)
    setGrade(event.grade)
    setSimilarity(event.similarity)
    setTimingOffset(event.timing_offset_ms ?? null)
    setHoldState(event.hold_state || null)
    setTrackingLost(Boolean(event.tracking_lost))
    setLimbScores(event.limb_scores || {})
    drawCoachPose(event.coach_pose)
    if (gradeTimeoutRef.current) clearTimeout(gradeTimeoutRef.current)
    gradeTimeoutRef.current = setTimeout(() => setGrade(''), 650)
  }, [drawCoachPose])

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
    return () => { ws.close(); if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [sessionId, handleScoreEvent])

  const clockLoop = useCallback((now: number) => {
    const video = videoRef.current
    const ws = wsRef.current
    if (!video || !ws || ws.readyState !== WebSocket.OPEN || video.paused || video.ended) {
      rafRef.current = requestAnimationFrame(clockLoop)
      return
    }
    const mediaMs = Math.max(0, Math.round(video.currentTime * 1000))
    setProgress(Math.min(100, mediaMs / Math.max(1, playback?.duration_ms || detail?.duration_ms || 1) * 100))

    if (now - lastClockSentRef.current >= 45) {
      ws.send(JSON.stringify({ action: 'media_clock', media_time_ms: mediaMs }))
      lastClockSentRef.current = now
    }

    while (beatIndexRef.current < beats.length && mediaMs >= beats[beatIndexRef.current]) {
      if (mediaMs - beats[beatIndexRef.current] < 150) setBeatPulse(v => v + 1)
      beatIndexRef.current += 1
    }
    rafRef.current = requestAnimationFrame(clockLoop)
  }, [beats, playback, detail])

  const startGame = async () => {
    if (!videoRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
    for (const value of ['3', '2', '1']) {
      setCountdown(value)
      await new Promise(resolve => setTimeout(resolve, 650))
    }
    setCountdown('GO!')
    await new Promise(resolve => setTimeout(resolve, 350))
    setCountdown(null)
    videoRef.current.currentTime = 0
    videoRef.current.muted = false
    await videoRef.current.play()
    wsRef.current.send(JSON.stringify({ action: 'start', media_time_ms: 0 }))
    setStarted(true)
    beatIndexRef.current = 0
    lastClockSentRef.current = 0
    rafRef.current = requestAnimationFrame(clockLoop)
  }

  const togglePause = async () => {
    const video = videoRef.current
    const ws = wsRef.current
    if (!video || !ws || ws.readyState !== WebSocket.OPEN) return
    if (video.paused) {
      await video.play()
      ws.send(JSON.stringify({ action: 'resume' }))
      setPaused(false)
    } else {
      video.pause()
      ws.send(JSON.stringify({ action: 'pause' }))
      setPaused(true)
    }
  }

  const stopGame = () => {
    videoRef.current?.pause()
    wsRef.current?.send(JSON.stringify({ action: 'stop' }))
  }

  const gradeClass = (g: string) => g ? `grade-${g}` : ''
  const timingText = timingOffset == null ? '' : timingOffset < 45 ? 'ON BEAT' : timingOffset < 90 ? 'CLOSE' : `${timingOffset}ms`

  if (gameOver && results) {
    const hits = Object.values(results.grade_counts || {}).reduce((a: number, b: any) => a + Number(b || 0), 0) as number
    const quality = hits ? ((results.grade_counts?.perfect || 0) + .85 * (results.grade_counts?.super || 0) + .65 * (results.grade_counts?.good || 0) + .4 * (results.grade_counts?.ok || 0)) / hits : 0
    const stars = Math.max(1, Math.min(5, Math.round(quality * 5)))
    return (
      <div className="min-h-screen dance-shell p-6 flex items-center justify-center overflow-auto">
        <div className="glass-card max-w-3xl w-full rounded-[2rem] p-8 md:p-10 text-center">
          <p className="dance-kicker">ROUTINE COMPLETE</p>
          <div className="text-7xl md:text-8xl font-black mt-2 score-gradient">{Number(results.total_score || 0).toLocaleString()}</div>
          <div className="text-4xl tracking-[.3em] mt-4">{'★'.repeat(stars)}<span className="text-white/15">{'★'.repeat(5-stars)}</span></div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-8">
            <Stat value={`${results.max_combo || 0}x`} label="Max combo" />
            <Stat value={`${Math.round((results.accuracy_arms || 0) * 100)}%`} label="Arms" />
            <Stat value={`${Math.round((results.accuracy_legs || 0) * 100)}%`} label="Legs" />
            <Stat value={`${Math.round((results.accuracy_torso || 0) * 100)}%`} label="Torso" />
          </div>
          <div className="grid grid-cols-5 gap-2 mt-6">
            {['perfect','super','good','ok','x'].map(g => <div key={g} className="rounded-2xl bg-white/5 p-3"><div className={`font-black uppercase ${gradeClass(g)}`}>{g}</div><div className="text-2xl font-black mt-1">{results.grade_counts?.[g] || 0}</div></div>)}
          </div>
          <div className="grid md:grid-cols-2 gap-3 mt-8">
            <button className="dance-primary rounded-2xl py-4 font-black" onClick={() => navigate(`/connect/${danceId}`)}>Dance again</button>
            <button className="rounded-2xl py-4 font-black bg-white/10 hover:bg-white/15" onClick={() => navigate('/')}>Library</button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="game-stage">
      <div key={beatPulse} className="beat-flash" />
      <div className="stage-orb orb-one" /><div className="stage-orb orb-two" />
      {danceId && (
        <video
          ref={videoRef}
          src={videoUrl(danceId)}
          className="coach-video"
          style={{ transform: detail?.mirror_mode ? 'scaleX(-1)' : undefined }}
          playsInline
          preload="auto"
          onEnded={() => wsRef.current?.send(JSON.stringify({ action: 'media_clock', media_time_ms: playback?.duration_ms || detail?.duration_ms || 0 }))}
        />
      )}
      <canvas ref={canvasRef} className="coach-pose-overlay" />
      <div className="video-vignette" />

      <div className="game-hud top-hud">
        <div><div className="hud-label">SCORE</div><div className="hud-score">{score.toLocaleString()}</div></div>
        <div className="hud-center"><div className="song-title">{detail?.title || 'Loading routine…'}</div><div className="progress-track"><div className="progress-fill" style={{ width: `${progress}%` }} /></div></div>
        <div className="text-right"><div className="hud-label">COMBO</div><div className={`hud-score ${combo > 2 ? 'combo-hot' : ''}`}>{combo}x</div></div>
      </div>

      <div className="limb-meter left-meter"><LimbMeter label="ARMS" value={limbScores.arms || 0} /><LimbMeter label="TORSO" value={limbScores.torso || 0} /><LimbMeter label="LEGS" value={limbScores.legs || 0} /></div>

      {grade && !trackingLost && <div className="grade-burst"><div className={`grade-word ${gradeClass(grade)}`}>{grade === 'x' ? 'X' : grade}</div><div className="timing-chip">{timingText}</div></div>}
      {trackingLost && <div className="tracking-warning"><div className="text-3xl font-black">STEP BACK INTO FRAME</div><div className="text-sm text-white/65 mt-1">No points are awarded while tracking is lost.</div></div>}
      {!phoneConnected && <div className="phone-warning">Phone disconnected — reconnect the tracker to continue scoring</div>}
      {holdState && ['entering','holding'].includes(holdState) && <div className="hold-burst">HOLD!</div>}
      {holdState === 'yeah' && <div className="yeah-burst">YEAH!</div>}

      {countdown && (
        <div className="start-overlay">
          <div className="start-panel">
            <div className="countdown-word">{countdown}</div>
            {countdown === 'READY' && <>
              <div className="text-white/55 mt-3">Phone tracking is ready. Audio starts on your click.</div>
              <button disabled={!ready || !detail || !playback} onClick={startGame} className="dance-primary px-10 py-4 rounded-2xl font-black text-xl mt-7 disabled:opacity-30">START DANCE</button>
            </>}
          </div>
        </div>
      )}

      {started && !gameOver && (
        <div className="game-controls">
          <button onClick={togglePause}>{paused ? '▶ Resume' : 'Ⅱ Pause'}</button>
          <button onClick={stopGame}>■ Stop</button>
        </div>
      )}

      <div className="latency-strip"><span>{Math.round(similarity * 100)}% pose</span><span>•</span><span>{timingText || 'syncing'}</span></div>
    </div>
  )
}

function Stat({value, label}: {value: string, label: string}) {
  return <div className="rounded-2xl bg-white/5 p-4"><div className="text-3xl font-black">{value}</div><div className="text-xs text-white/45 uppercase tracking-widest mt-1">{label}</div></div>
}

function LimbMeter({label, value}: {label: string, value: number}) {
  return <div className="limb-row"><span>{label}</span><div className="limb-track"><div className="limb-fill" style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} /></div></div>
}
