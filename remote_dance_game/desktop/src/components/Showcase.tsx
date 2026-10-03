import React, { useEffect, useRef, useState } from 'react'

function useElapsed() {
  const [ms, setMs] = useState(0)
  const start = useRef(performance.now())
  useEffect(() => {
    const timer = window.setInterval(() => setMs(performance.now() - start.current), 50)
    return () => window.clearInterval(timer)
  }, [])
  return ms
}

export default function Showcase() {
  const ms = useElapsed()
  const stage = ms < 5200 ? 0 : ms < 9800 ? 1 : ms < 15800 ? 2 : ms < 19000 ? 3 : 4
  const progress = Math.min(100, Math.round(ms / 48))
  const score = Math.max(0, Math.round((ms - 19000) * 1.72))
  const combo = Math.max(0, Math.min(28, Math.floor((ms - 19000) / 330)))
  const grades = ['GOOD', 'SUPER', 'PERFECT', 'SUPER', 'PERFECT']
  const grade = grades[Math.max(0, Math.floor((ms - 19000) / 1500)) % grades.length]

  if (stage === 0) {
    return <main className="showcase builder-screen">
      <div className="builder-aurora" />
      <header className="builder-header"><div className="builder-back">←</div><div className="tv-logo">DANCE<span>FLOW</span></div><div className="source-tabs"><button className="active">FILE</button><button>LINK</button></div></header>
      <section className="builder-main showcase-builder">
        <div className="builder-copy"><div className="active-eyebrow">CREATE A DANCE</div><h1>Drop in a video.<br/>Get a playable stage.</h1><p>Movement, timing, colors and effects are generated automatically.</p></div>
        <div className="video-drop has-file"><video src="/showcase/source.mp4" autoPlay muted loop playsInline/><div className="drop-wash"/><div className="drop-content"><div className="drop-plus">✓</div><strong>night_move.mp4</strong><span>ready to transform</span></div></div>
        <div className="builder-bottom-row"><div className="title-ghost fake">Night Move</div><div className="tv-play build-action fake-button">GENERATE DANCE →</div></div>
      </section>
      {ms > 2200 && <div className="showcase-process-overlay"><div className="processing-ring mini" style={{ '--progress': `${progress * 3.6}deg` } as React.CSSProperties}><div><strong>{progress}%</strong><span>{progress < 45 ? 'Learning the choreography' : progress < 78 ? 'Finding the beat' : 'Painting the game stage'}</span></div></div></div>}
    </main>
  }

  if (stage === 1) {
    return <main className="showcase-result">
      <div className="tv-logo showcase-logo">DANCE<span>FLOW</span></div>
      <div className="before-after">
        <div className="compare-panel"><video src="/showcase/source.mp4" autoPlay muted loop playsInline/><span>YOUR VIDEO</span></div>
        <div className="transform-arrow">→</div>
        <div className="compare-panel result"><video src="/showcase/game.mp4" autoPlay muted loop playsInline/><span>PLAYABLE STAGE</span></div>
      </div>
      <div className="showcase-caption"><b>CHARACTER · STAGE · BEAT EFFECTS</b><span>generated automatically</span></div>
    </main>
  }

  if (stage === 2) {
    return <main className="showcase tv-library" style={{'--theme-primary':'rgb(77,122,255)','--theme-secondary':'rgb(255,77,222)','--theme-accent':'rgb(255,244,117)'} as React.CSSProperties}>
      <div className="library-backdrop"><video src="/showcase/game.mp4" className="library-preview" autoPlay muted loop playsInline/><div className="library-wash"/><div className="library-color-field"/></div>
      <header className="tv-header"><div className="tv-logo">DANCE<span>FLOW</span></div><div className="tv-header-actions"><div className="tv-icon">＋</div><div className="tv-icon tv-icon-muted">⚙</div></div></header>
      <section className="active-copy"><div className="active-eyebrow">MIDNIGHT POP</div><h1>Night Move</h1><div className="active-meta"><span>2:41</span><span>•</span><span>medium</span></div><div className="active-actions"><div className="tv-play">▶ PLAY</div><div className="tv-more">•••</div></div></section>
      <section className="carousel-shell"><div className="carousel-hint">YOUR DANCES</div><div className="dance-rail showcase-rail"><div className="dance-tile active"><div className="tile-art"><video src="/showcase/game.mp4" autoPlay muted loop playsInline/><div className="tile-shade"/><div className="tile-play">▶</div></div><div className="tile-title">Night Move</div></div><div className="dance-tile ghost-card"><div className="tile-art"/><div className="tile-title">Electric Bloom</div></div><div className="dance-tile ghost-card alt"><div className="tile-art"/><div className="tile-title">Afterglow</div></div><div className="dance-tile add-tile"><div className="add-circle">＋</div><div className="tile-title">Add dance</div></div></div></section>
    </main>
  }

  if (stage === 3) {
    return <main className="showcase connect-screen"><div className="connect-blob blob-a"/><div className="connect-blob blob-b"/><div className="connect-copy"><div className="active-eyebrow">ONE QUICK STEP</div><h1>Point your phone<br/>at the room.</h1><div className="connect-statuses"><div className="ready"><b>✓</b><span>Phone connected</span></div><div className="ready"><b>✓</b><span>Full body visible</span></div></div><div className="tv-play connect-play">START DANCE →</div></div><div className="qr-stage"><div className="phone-frame demo-phone"><div className="phone-notch"/><div className="demo-qr">▦</div><div className="phone-caption">READY</div></div></div></main>
  }

  return <main className="showcase game-stage showcase-game">
    <video src="/showcase/game.mp4" className="coach-video" autoPlay muted loop playsInline/>
    <div className="video-vignette"/>
    <div className="game-hud top-hud"><div><div className="hud-label">SCORE</div><div className="hud-score">{score.toLocaleString()}</div></div><div className="hud-center"><div className="song-title">Night Move</div><div className="progress-track"><div className="progress-fill" style={{width:`${Math.min(100,(ms-19000)/100)}%`}}/></div></div><div className="text-right"><div className="hud-label">COMBO</div><div className="hud-score combo-hot">{combo}x</div></div></div>
    <div className="grade-burst demo-grade"><div className={`grade-word grade-${grade.toLowerCase()}`}>{grade}</div><div className="timing-chip">ON BEAT</div></div>
    <div className="showcase-phone-pill">● PHONE · LIVE</div>
  </main>
}
