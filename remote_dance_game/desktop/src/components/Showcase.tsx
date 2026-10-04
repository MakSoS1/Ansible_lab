import React, { useEffect, useState } from 'react'
import PoseGlyph from './PoseGlyph'

export default function Showcase() {
  const [ms, setMs] = useState(0)
  useEffect(() => {
    const start = performance.now()
    const id = window.setInterval(() => setMs(performance.now() - start), 50)
    return () => window.clearInterval(id)
  }, [])

  const stage = ms < 5200 ? 0 : ms < 9000 ? 1 : ms < 17200 ? 2 : ms < 20800 ? 3 : ms < 24500 ? 4 : 5
  const gameMs = Math.max(0, ms - 24500)
  const score = 28450 + Math.floor(gameMs * 4.0)
  const moveNo = Math.floor(gameMs / 520)
  const combo = Math.min(64, 7 + moveNo)
  const progress = Math.min(100, 34 + gameMs / 155)
  const grade = ['PERFECT','SUPER','GOOD','PERFECT','SUPER','OK'][moveNo % 6]

  if (stage === 0) {
    const progress = Math.min(100, Math.floor(Math.max(0, ms - 1500) / 33))
    return <main className="ref-builder-demo">
      <div className="ref-builder-demo-bg"><video src="/showcase/source.webm" autoPlay muted loop playsInline /></div>
      <header className="ref-connect-header"><div className="ref-brand compact">DANCE<span>FLOW</span></div><div className="ref-tabs"><button className="active">File</button><button>Link</button></div></header>
      <section className="ref-builder-demo-grid">
        <div><p>CREATE A DANCE</p><h1>Drop in a video.<br/>Get a playable stage.</h1><span>Movement, timing, character treatment and the visual stage are generated automatically.</span><div className="ref-primary fake">GENERATE DANCE →</div></div>
        <div className="ref-builder-video"><video src="/showcase/source.webm" autoPlay muted loop playsInline/><b>night_move.mp4</b><span>ready to transform</span></div>
      </section>
      {ms > 1600 && <div className="ref-processing"><strong>{progress}%</strong><span>{progress < 45 ? 'Learning choreography' : progress < 75 ? 'Finding the beat' : 'Painting the stage'}</span><i><b style={{width:`${progress}%`}} /></i></div>}
    </main>
  }

  if (stage === 1) {
    return <main className="ref-compare">
      <div className="ref-brand compact">DANCE<span>FLOW</span></div>
      <h1>Source → playable stage</h1>
      <div className="ref-compare-grid">
        <figure><video src="/showcase/source.webm" autoPlay muted loop playsInline/><figcaption>YOUR VIDEO</figcaption></figure>
        <b>→</b>
        <figure className="after"><video src="/showcase/game.webm" autoPlay muted loop playsInline/><figcaption>PLAYABLE STAGE</figcaption></figure>
      </div>
      <p>Character · clear background · beat-reactive stage</p>
    </main>
  }

  if (stage === 2) {
    return <main className="ref-library showcase-reference-menu" style={{'--theme-primary':'rgb(70,220,255)','--theme-secondary':'rgb(255,52,194)','--theme-accent':'rgb(255,226,70)'} as React.CSSProperties}>
      <div className="ref-library-backdrop"><video src="/showcase/game.webm" className="ref-library-preview" autoPlay muted loop playsInline/><div className="ref-library-blur"/><div className="ref-library-vignette"/></div>
      <header className="ref-topbar">
        <div className="ref-brand">DANCE<span>FLOW</span><small>PLAY · MOVE · TOGETHER</small></div>
        <div className="ref-tab-shell"><span className="ref-bumper">L1</span><nav className="ref-tabs"><button className="active">Songs</button><button>Playlists</button><button>Challenges</button><button>Party</button></nav><span className="ref-bumper">R1</span></div>
        <div className="ref-profile"><div className="ref-avatar">D</div><div className="ref-profile-copy"><strong>Player</strong><span>Lv. 12</span><i><b /></i></div><div className="ref-points">★ <strong>1,260</strong></div><div className="ref-settings">⚙</div></div>
      </header>
      <aside className="ref-slogan"><span>FEEL</span><span>THE BEAT</span><span>ANYWHERE</span><b/></aside>
      <aside className="ref-graffiti">GOOD<br/>DANCE<br/><em>BRIGHTER</em><br/>PEOPLE</aside>
      <section className="ref-carousel-stage">
        <div className="ref-arrow ref-arrow-left">‹</div>
        <div className="ref-card-rail showcase-cards">
          {[-2,-1,0,1,2].map((offset,i)=><div key={offset} className={`ref-song-card ${offset===0?'active':''}`} style={{'--offset':offset,zIndex:20-Math.abs(offset)} as React.CSSProperties}><div className="ref-card-art"><video src="/showcase/game.webm" autoPlay muted loop playsInline/><div className="ref-card-shade"/><div className="ref-card-title-brush">{['Neon Hearts','Better Together','Night Move','Stardust','Feel Alive'][i]}</div></div><div className="ref-card-footer"><span className="ref-difficulty-bars"><i/><i/><i/><i/></span><span>Medium</span><span>3:12</span><b>Solo</b></div></div>)}
        </div>
        <div className="ref-arrow ref-arrow-right">›</div>
      </section>
      <section className="ref-selected-copy"><h1>Night Move</h1><div>DANCEFLOW CREW <span>│</span> POP <span>│</span> 2026</div><div className="ref-selected-actions"><div className="ref-primary">▶ <strong>Play</strong></div><div className="ref-secondary">♥ Favorite</div><div className="ref-secondary">◉ Preview</div></div></section>
      <div className="ref-bottom-hints"><span>◉ Back</span><span>△ Options</span></div>
    </main>
  }

  if (stage === 3) {
    return <main className="ref-connect showcase-connect">
      <img className="ref-connect-bg" src="/showcase/poster.jpg"/><div className="ref-connect-wash"/>
      <header className="ref-connect-header"><div className="ref-brand compact">DANCE<span>FLOW</span></div></header>
      <div className="ref-connect-grid"><section className="ref-connect-copy"><p>ONE QUICK STEP</p><h1>Point your phone<br/>at the room.</h1><span>Keep your full body visible. The phone only sends pose points.</span><div className="ref-connect-statuses"><div className="ready"><b>✓</b><span>Phone connected</span></div><div className="ready"><b>✓</b><span>Full body visible</span></div></div><div className="ref-primary ref-connect-play">START DANCE →</div></section><section className="ref-phone-stage"><div className="ref-phone-frame"><div className="ref-phone-notch"/><div className="demo-qr">▦</div><strong>READY</strong><span>Night Move</span></div></section></div>
    </main>
  }

  if (stage === 4) {
    return <main className="showcase-coach-select">
      <div className="showcase-coach-bg"><video src="/showcase/game.webm" autoPlay muted loop playsInline/></div>
      <div className="showcase-coach-wash"/>
      <header><div className="ref-brand compact">DANCE<span>FLOW</span></div><span>PLAYER 1</span></header>
      <h1>SELECT YOUR COACH</h1>
      <p>Choose the dancer you want to follow.</p>
      <section>
        <article className="active"><video src="/showcase/game.webm" autoPlay muted loop playsInline/><b>COACH 1</b><em>✓</em></article>
        <article><video src="/showcase/game.webm" autoPlay muted loop playsInline/><b>COACH 2</b></article>
      </section>
      <footer>◀  ▶  CHOOSE &nbsp;&nbsp;&nbsp;&nbsp; A / ENTER  CONFIRM</footer>
    </main>
  }

  return <main className="ref-gameplay showcase-reference-game">
    <video src="/showcase/game.webm" className="ref-game-video" autoPlay muted loop playsInline/>
    <div className="ref-game-vignette"/>
    <section className="ref-song-panel"><div className="ref-song-cover"><img src="/showcase/poster.jpg"/></div><div className="ref-song-copy"><h1>Night Move</h1><p>DANCEFLOW CREW <i>│</i> POP <i>│</i> 2026</p><div className="ref-song-progress"><b style={{width:`${progress}%`}}/><em/></div></div><div className="ref-song-time">1:28 / 3:12</div></section>
    <section className="showcase-player-huds">
      {[0,1].map((player)=><div key={player} className={`showcase-player-card player-${player}`}>
        <div className="showcase-live-mirror"><PoseGlyph active variant={moveNo+player}/></div>
        <div className="showcase-player-copy"><span>P{player+1} · COACH {player+1}</span><b key={`${moveNo}-${player}`}>{player===0 ? grade : ['PERFECT','GOOD','SUPER'][moveNo%3]}</b><small>{(score-player*4200).toLocaleString()} · COMBO {Math.max(1,combo-player*2)}</small></div>
      </div>)}
    </section>
    <aside className="showcase-cues">
      {[0,1].map((player)=><div key={player} className={`showcase-cue player-${player}`}><PoseGlyph active variant={moveNo+player+2}/><i><b style={{width:`${25+((moveNo*23+player*17)%70)}%`}}/></i></div>)}
    </aside>
    <section className="ref-timeline"><div className="ref-wave">{Array.from({length:13},(_,i)=><i key={i} style={{height:`${10+((i*13)%30)}px`}}/>)}</div><div className="ref-timeline-track"><div className="ref-timeline-fill" style={{width:`${progress}%`}}/>{[12,30,49,68,87].map((p,i)=><i key={i} className="ref-timeline-marker" style={{left:`${p}%`}}/>)}<b className="ref-timeline-cursor" style={{left:`${progress}%`}}/></div><div className="ref-wave">{Array.from({length:13},(_,i)=><i key={i} style={{height:`${10+(((12-i)*13)%30)}px`}}/>)}</div></section>
  </main>
}

function gradeClass(g: string) { return `grade-${g.toLowerCase()}` }
