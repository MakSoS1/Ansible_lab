import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { apiDelete, apiGet, Dance, posterUrl, videoUrl } from '../api'

function fmt(ms: number) {
  const sec = Math.max(0, Math.floor(ms / 1000))
  return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')}`
}

function themeStyle(dance?: Dance): React.CSSProperties {
  const p = dance?.theme?.primary || [74, 220, 255]
  const s = dance?.theme?.secondary || [255, 52, 194]
  const a = dance?.theme?.accent || [255, 226, 70]
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

  const active = dances[selected]

  const load = async () => {
    try {
      setLoading(true)
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
    const timer = window.setTimeout(() => setPreviewReady(true), 720)
    return () => window.clearTimeout(timer)
  }, [active?.dance_id])

  useEffect(() => {
    const video = previewRef.current
    if (!video || !previewReady) return
    video.currentTime = 0
    video.volume = hasGesture ? 0.09 : 0
    video.muted = !hasGesture
    video.play().catch(() => {
      video.muted = true
      video.play().catch(() => {})
    })
  }, [previewReady, active?.dance_id, hasGesture])

  const choose = (delta: number) => {
    if (!dances.length) return
    setHasGesture(true)
    setSelected(i => Math.max(0, Math.min(dances.length - 1, i + delta)))
  }

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!dances.length) return
      if (event.key === 'ArrowRight') { event.preventDefault(); choose(1) }
      else if (event.key === 'ArrowLeft') { event.preventDefault(); choose(-1) }
      else if (event.key === 'Enter' && active) {
        event.preventDefault(); setHasGesture(true); navigate(`/connect/${active.dance_id}`)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [dances.length, active?.dance_id])

  const remove = async () => {
    if (!active || !confirm(`Delete “${active.title}”?`)) return
    await apiDelete(`/api/dances/${active.dance_id}`)
    await load()
    setSelected(i => Math.max(0, Math.min(i, dances.length - 2)))
  }

  const background = useMemo(() => active?.has_video ? videoUrl(active.dance_id) : '', [active?.dance_id])

  if (loading) return <div className="ref-loader"><div className="ref-brand">DANCE<span>FLOW</span><small>PLAY · MOVE · TOGETHER</small></div></div>

  if (!dances.length) {
    return (
      <main className="ref-empty" onPointerDown={() => setHasGesture(true)}>
        <div className="ref-brand">DANCE<span>FLOW</span><small>PLAY · MOVE · TOGETHER</small></div>
        <h1>Your dance shelf is empty.</h1>
        <p>Add one video and DanceFlow will build the choreography, stage and playable routine.</p>
        <button className="ref-primary" onClick={() => navigate('/build')}>＋ Add a dance</button>
        {error && <div className="tv-error">{error}</div>}
      </main>
    )
  }

  return (
    <main className="ref-library" style={themeStyle(active)} onPointerDown={() => setHasGesture(true)}>
      <div className="ref-library-backdrop">
        {background && previewReady && (
          <video key={active.dance_id} ref={previewRef} src={background} className="ref-library-preview" loop playsInline preload="metadata" />
        )}
        {!previewReady && active?.has_poster && <img src={posterUrl(active.dance_id)} className="ref-library-preview ref-library-poster" />}
        <div className="ref-library-blur" />
        <div className="ref-library-vignette" />
      </div>

      <header className="ref-topbar">
        <div className="ref-brand">DANCE<span>FLOW</span><small>PLAY · MOVE · TOGETHER</small></div>
        <nav className="ref-tabs" aria-label="Game sections">
          <button className="active">Songs</button><button>Playlists</button><button>Challenges</button><button>Party</button>
        </nav>
        <div className="ref-profile">
          <div className="ref-avatar">D</div>
          <div className="ref-profile-copy"><strong>Player</strong><span>Lv. 12</span><i><b /></i></div>
          <div className="ref-points">★ <strong>1,260</strong></div>
          <button className="ref-settings" aria-label="Settings" onClick={() => navigate('/settings')}>⚙</button>
        </div>
      </header>

      <aside className="ref-slogan"><span>FEEL</span><span>THE BEAT</span><span>ANYWHERE</span><b /></aside>
      <aside className="ref-graffiti">GOOD<br/>DANCE<br/><em>BRIGHTER</em><br/>PEOPLE</aside>

      <section className="ref-carousel-stage">
        <button className="ref-arrow ref-arrow-left" onClick={() => choose(-1)} aria-label="Previous dance">‹</button>
        <div className="ref-card-rail">
          {dances.map((dance, index) => {
            const offset = index - selected
            if (Math.abs(offset) > 3) return null
            return (
              <button
                key={dance.dance_id}
                className={`ref-song-card ${offset === 0 ? 'active' : ''}`}
                style={{
                  ...themeStyle(dance),
                  '--offset': offset,
                  zIndex: 20 - Math.abs(offset),
                } as React.CSSProperties}
                onMouseEnter={() => setSelected(index)}
                onFocus={() => setSelected(index)}
                onClick={() => offset === 0 ? navigate(`/connect/${dance.dance_id}`) : setSelected(index)}
              >
                <div className="ref-card-art">
                  {dance.has_poster ? <img src={posterUrl(dance.dance_id)} /> : <div className="ref-card-fallback" />}
                  <div className="ref-card-shade" />
                  <div className="ref-card-title-brush">{dance.title}</div>
                </div>
                <div className="ref-card-footer">
                  <span className="ref-difficulty-bars"><i/><i/><i/><i/></span>
                  <span>{dance.difficulty}</span><span>{fmt(dance.duration_ms)}</span><b>Solo</b>
                </div>
              </button>
            )
          })}
        </div>
        <button className="ref-arrow ref-arrow-right" onClick={() => choose(1)} aria-label="Next dance">›</button>
      </section>

      <section className="ref-selected-copy">
        <h1>{active.title}</h1>
        <div>{active.theme?.name || 'DANCEFLOW CREW'} <span>│</span> POP <span>│</span> PLAYABLE</div>
        <div className="ref-selected-actions">
          <button className="ref-primary" onClick={() => navigate(`/connect/${active.dance_id}`)}>▶ <strong>Play</strong></button>
          <button className="ref-secondary">♥ <span>Favorite</span></button>
          <button className="ref-secondary" onClick={() => setPreviewReady(true)}>◉ <span>Preview</span></button>
        </div>
      </section>

      <div className="ref-bottom-hints"><span>◉ Back</span><span>△ Options</span></div>
      <button className="ref-add-dance" aria-label="Add dance" onClick={() => navigate('/build')}>＋</button>
      <button className="ref-delete-dance" title="Delete selected dance" onClick={remove}>•••</button>
      {error && <div className="tv-error">{error}</div>}
    </main>
  )
}
