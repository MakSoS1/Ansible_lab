import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { apiDelete, apiGet, Dance, posterUrl, videoUrl } from '../api'

function fmt(ms: number) {
  const sec = Math.max(0, Math.floor(ms / 1000))
  return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')}`
}

function themeStyle(dance?: Dance): React.CSSProperties {
  const p = dance?.theme?.primary || [76, 232, 255]
  const s = dance?.theme?.secondary || [255, 67, 183]
  const a = dance?.theme?.accent || [255, 232, 91]
  return {
    '--theme-primary': `rgb(${p.join(',')})`,
    '--theme-secondary': `rgb(${s.join(',')})`,
    '--theme-accent': `rgb(${a.join(',')})`,
  } as React.CSSProperties
}

export default function Library() {
  const [dances, setDances] = useState<Dance[]>([])
  const [selected, setSelected] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [previewReady, setPreviewReady] = useState(false)
  const [hasGesture, setHasGesture] = useState(false)
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const previewRef = useRef<HTMLVideoElement | null>(null)
  const railRef = useRef<HTMLDivElement | null>(null)

  const active = dances[selected]

  const load = async () => {
    try {
      const data: Dance[] = await apiGet('/api/dances')
      setDances(data)
      const focus = searchParams.get('focus')
      if (focus) {
        const idx = data.findIndex(d => d.dance_id === focus)
        if (idx >= 0) setSelected(idx)
      }
      setError('')
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  useEffect(() => {
    setPreviewReady(false)
    if (!active?.has_video) return
    const timer = window.setTimeout(() => setPreviewReady(true), 850)
    return () => window.clearTimeout(timer)
  }, [active?.dance_id])

  useEffect(() => {
    const video = previewRef.current
    if (!video || !previewReady) return
    video.currentTime = 0
    video.volume = hasGesture ? 0.12 : 0
    video.muted = !hasGesture
    video.play().catch(() => {
      video.muted = true
      video.play().catch(() => {})
    })
  }, [previewReady, active?.dance_id, hasGesture])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!dances.length) return
      if (event.key === 'ArrowRight') {
        event.preventDefault(); setHasGesture(true); setSelected(i => Math.min(dances.length - 1, i + 1))
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault(); setHasGesture(true); setSelected(i => Math.max(0, i - 1))
      } else if (event.key === 'Enter' && active) {
        event.preventDefault(); setHasGesture(true); navigate(`/connect/${active.dance_id}`)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [dances.length, active?.dance_id])

  useEffect(() => {
    const card = railRef.current?.querySelector(`[data-index="${selected}"]`) as HTMLElement | null
    card?.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' })
  }, [selected])

  const background = useMemo(() => active?.has_video ? videoUrl(active.dance_id) : '', [active?.dance_id])

  const remove = async () => {
    if (!active || !confirm(`Delete “${active.title}”?`)) return
    await apiDelete(`/api/dances/${active.dance_id}`)
    await load()
    setSelected(i => Math.max(0, Math.min(i, dances.length - 2)))
  }

  if (loading) return <div className="tv-loader"><div className="tv-logo">DANCE<span>FLOW</span></div></div>

  if (!dances.length) {
    return (
      <div className="tv-empty" onPointerDown={() => setHasGesture(true)}>
        <div className="tv-logo">DANCE<span>FLOW</span></div>
        <div className="empty-orb" />
        <h1>Your dance shelf is empty.</h1>
        <p>Drop in a video. The game will learn the choreography and build the stage automatically.</p>
        <button className="tv-play" onClick={() => navigate('/build')}>＋ ADD A DANCE</button>
        {error && <div className="tv-error">{error}</div>}
      </div>
    )
  }

  return (
    <main className="tv-library" style={themeStyle(active)} onPointerDown={() => setHasGesture(true)}>
      <div className="library-backdrop">
        {background && previewReady && (
          <video key={active.dance_id} ref={previewRef} src={background} className="library-preview" loop playsInline preload="metadata" />
        )}
        {!previewReady && active?.has_poster && <img src={posterUrl(active.dance_id)} className="library-poster-bg" />}
        <div className="library-wash" />
        <div className="library-color-field" />
      </div>

      <header className="tv-header">
        <div className="tv-logo">DANCE<span>FLOW</span></div>
        <div className="tv-header-actions">
          <button aria-label="Add dance" className="tv-icon" onClick={() => navigate('/build')}>＋</button>
          <button aria-label="Settings" className="tv-icon tv-icon-muted" onClick={() => navigate('/settings')}>⚙</button>
        </div>
      </header>

      <section className="active-copy">
        <div className="active-eyebrow">{active.theme?.name || 'PLAYABLE ROUTINE'}</div>
        <h1>{active.title}</h1>
        <div className="active-meta"><span>{fmt(active.duration_ms)}</span><span>•</span><span>{active.difficulty}</span></div>
        <div className="active-actions">
          <button className="tv-play" onClick={() => navigate(`/connect/${active.dance_id}`)}>▶ PLAY</button>
          <button className="tv-more" title="Delete" onClick={remove}>•••</button>
        </div>
      </section>

      <section className="carousel-shell">
        <div className="carousel-hint">YOUR DANCES</div>
        <div className="dance-rail" ref={railRef} onWheel={e => {
          if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
            e.preventDefault()
            setHasGesture(true)
            setSelected(i => Math.max(0, Math.min(dances.length - 1, i + (e.deltaY > 0 ? 1 : -1))))
          }
        }}>
          {dances.map((dance, index) => (
            <button
              key={dance.dance_id}
              data-index={index}
              className={`dance-tile ${index === selected ? 'active' : ''}`}
              onMouseEnter={() => setSelected(index)}
              onFocus={() => setSelected(index)}
              onClick={() => index === selected ? navigate(`/connect/${dance.dance_id}`) : setSelected(index)}
              style={themeStyle(dance)}
            >
              <div className="tile-art">
                {dance.has_poster ? <img src={posterUrl(dance.dance_id)} /> : <div className="tile-fallback" />}
                <div className="tile-shade" />
                <div className="tile-play">▶</div>
              </div>
              <div className="tile-title">{dance.title}</div>
            </button>
          ))}
          <button className="dance-tile add-tile" onClick={() => navigate('/build')}>
            <div className="add-circle">＋</div><div className="tile-title">Add dance</div>
          </button>
        </div>
      </section>

      <div className="tv-nav-hint"><span>← →</span> choose <span>ENTER</span> play</div>
      {error && <div className="tv-error">{error}</div>}
    </main>
  )
}
