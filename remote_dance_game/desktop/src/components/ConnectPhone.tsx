import React, { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { QRCodeSVG } from 'qrcode.react'
import { apiGet, apiPost, GameSession } from '../api'

export default function ConnectPhone() {
  const { danceId } = useParams<{ danceId: string }>()
  const navigate = useNavigate()
  const [session, setSession] = useState<GameSession | null>(null)
  const [phoneConnected, setPhoneConnected] = useState(false)
  const [calibrated, setCalibrated] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!danceId) return
    apiPost('/api/session/create', { dance_id: danceId }).then(setSession).catch((e: any) => setError(e.message))
  }, [danceId])

  useEffect(() => {
    if (!session) return
    const timer = setInterval(async () => {
      try {
        const s = await apiGet(`/api/session/${session.session_id}/status`)
        setPhoneConnected(Boolean(s.phone_connected))
        setCalibrated(Boolean(s.calibrated))
      } catch (_) {}
    }, 800)
    return () => clearInterval(timer)
  }, [session?.session_id])

  return (
    <main className="connect-screen">
      <div className="connect-blob blob-a"/><div className="connect-blob blob-b"/>
      <button className="builder-back connect-back" onClick={() => navigate('/')}>←</button>
      <div className="connect-copy">
        <div className="active-eyebrow">ONE QUICK STEP</div>
        <h1>Point your phone<br/>at the room.</h1>
        <div className="connect-statuses">
          <div className={phoneConnected ? 'ready' : ''}><b>{phoneConnected ? '✓' : '1'}</b><span>Phone connected</span></div>
          <div className={calibrated ? 'ready' : ''}><b>{calibrated ? '✓' : '2'}</b><span>Full body visible</span></div>
        </div>
        <button disabled={!phoneConnected || !calibrated} className="tv-play connect-play" onClick={() => session && navigate(`/play/${danceId}/${session.session_id}`)}>
          {phoneConnected && calibrated ? 'START DANCE →' : 'WAITING FOR PHONE'}
        </button>
        {error && <div className="tv-error inline">{error}</div>}
      </div>
      <div className="qr-stage">
        <div className="phone-frame">
          <div className="phone-notch" />
          <div className="qr-card">
            {session ? <QRCodeSVG value={session.connect_url} size={310} bgColor="transparent" fgColor="#0b0b17" /> : <div className="qr-loading" />}
          </div>
          <div className="phone-caption">SCAN WITH CAMERA</div>
        </div>
      </div>
    </main>
  )
}
