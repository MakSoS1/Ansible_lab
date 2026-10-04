import React, { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { apiPost, apiGet, DanceDetail, GameSession, posterUrl } from '../api'
import { QRCodeSVG } from 'qrcode.react'

export default function ConnectPhone() {
  const { danceId } = useParams<{ danceId: string }>()
  const navigate = useNavigate()
  const [session, setSession] = useState<GameSession | null>(null)
  const [detail, setDetail] = useState<DanceDetail | null>(null)
  const [phoneConnected, setPhoneConnected] = useState(false)
  const [calibrated, setCalibrated] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!danceId) return
    Promise.all([
      apiPost('/api/session/create', { dance_id: danceId }),
      apiGet(`/api/dances/${danceId}`),
    ]).then(([s, d]) => {
      setSession(s)
      setDetail(d)
      setLoading(false)
    }).catch((e: any) => {
      setError(e.message)
      setLoading(false)
    })
  }, [danceId])

  useEffect(() => {
    if (!session) return
    const interval = setInterval(async () => {
      try {
        const status = await apiGet(`/api/session/${session.session_id}/status`)
        setPhoneConnected(Boolean(status.phone_connected))
        setCalibrated(Boolean(status.calibrated))
        if (status.phone_connected && status.calibrated) clearInterval(interval)
      } catch (_) {}
    }, 900)
    return () => clearInterval(interval)
  }, [session])

  return (
    <main className="ref-connect">
      {danceId && detail?.has_poster && <img className="ref-connect-bg" src={posterUrl(danceId)} />}
      <div className="ref-connect-wash" />
      <header className="ref-connect-header"><button onClick={() => navigate('/')}>←</button><div className="ref-brand compact">DANCE<span>FLOW</span></div></header>

      {loading ? <div className="ref-connect-loading">Creating session…</div> : error ? <div className="tv-error">{error}</div> : session && (
        <div className="ref-connect-grid">
          <section className="ref-connect-copy">
            <p>ONE QUICK STEP</p>
            <h1>Point your phone<br/>at the room.</h1>
            <span>Scan once, place the phone where your full body is visible, then leave it there while you dance.</span>
            <div className="ref-connect-statuses">
              <div className={phoneConnected ? 'ready' : ''}><b>{phoneConnected ? '✓' : '1'}</b><span>{phoneConnected ? 'Phone connected' : 'Scan the QR code'}</span></div>
              <div className={calibrated ? 'ready' : ''}><b>{calibrated ? '✓' : '2'}</b><span>{calibrated ? 'Full body visible' : 'Stand fully in frame'}</span></div>
            </div>
            <button className="ref-primary ref-connect-play" disabled={!phoneConnected || !calibrated} onClick={() => navigate(`/play/${danceId}/${session.session_id}`)}>
              {phoneConnected && calibrated ? 'START DANCE →' : 'WAITING FOR PHONE…'}
            </button>
          </section>
          <section className="ref-phone-stage">
            <div className="ref-phone-frame">
              <div className="ref-phone-notch" />
              <div className="ref-qr"><QRCodeSVG value={session.connect_url} size={260} bgColor="#ffffff" fgColor="#110b27" /></div>
              <strong>{calibrated ? 'READY' : phoneConnected ? 'CALIBRATING' : 'SCAN ME'}</strong>
              <span>{detail?.title || 'DanceFlow'}</span>
            </div>
          </section>
        </div>
      )}
    </main>
  )
}
