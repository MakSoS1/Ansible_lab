import React, { useEffect, useState } from 'react'

export default function App() {
  const [run, setRun] = useState(null)
  const [error, setError] = useState('')
  const packaged = run?.state === 'PACKAGED'

  async function start() {
    setError('')
    const created = await fetch('/api/runs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ seed: 42 }) }).then((r) => r.json())
    const detail = await fetch(`/api/runs/${created.id}`).then((r) => r.json())
    setRun({ ...detail, id: created.id })
  }

  useEffect(() => { fetch('/health').catch(() => setError('API offline')) }, [])

  return (
    <main style={{ fontFamily: 'Inter, sans-serif', margin: '2rem auto', maxWidth: 960 }}>
      <h1>AIOS Track 2</h1>
      <p>Surrogate bake-off, constraint guard, and contract NPV. Download stays disabled until the run is PACKAGED.</p>
      <button onClick={start}>Запустить pipeline</button>
      {error && <p>{error}</p>}
      {run && (
        <section>
          <p>State: {run.state}</p>
          <p>NPV: {run.npv_mrub}</p>
          <p>Backend: {run.backend}</p>
          <pre>{run.explanation}</pre>
          <ol>
            {(run.events || []).map((event, index) => (
              <li key={index}>{event.timestamp} — {event.actor} — {event.action}</li>
            ))}
          </ol>
          <a href={packaged ? `/api/runs/${run.id}/schedule` : undefined} aria-disabled={!packaged}>
            <button disabled={!packaged}>Скачать wells_schedule.inc</button>
          </a>
        </section>
      )}
    </main>
  )
}
