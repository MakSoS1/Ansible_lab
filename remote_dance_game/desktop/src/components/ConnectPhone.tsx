import React, { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { apiPost, apiGet, GameSession } from '../api'
import { QRCodeSVG } from 'qrcode.react'

export default function ConnectPhone() {
  const { danceId } = useParams<{ danceId: string }>()
  const navigate = useNavigate()
  const [session, setSession] = useState<GameSession | null>(null)
  const [phoneConnected, setPhoneConnected] = useState(false)
  const [calibrated, setCalibrated] = useState(false)
  const [sessionState, setSessionState] = useState('created')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!danceId) return
    apiPost('/api/session/create', { dance_id: danceId })
      .then((data: GameSession) => {
        setSession(data)
        setLoading(false)
      })
      .catch((e: any) => {
        setError(e.message)
        setLoading(false)
      })
  }, [danceId])

  useEffect(() => {
    if (!session) return
    const interval = setInterval(async () => {
      try {
        const status = await apiGet(`/api/session/${session.session_id}/status`)
        setSessionState(status.status || 'created')
        setPhoneConnected(Boolean(status.phone_connected))
        setCalibrated(Boolean(status.calibrated))
        if (status.phone_connected && status.calibrated) {
          clearInterval(interval)
        }
      } catch (e) {}
    }, 1000)
    return () => clearInterval(interval)
  }, [session])

  const handleStart = () => {
    if (session) {
      navigate(`/play/${danceId}/${session.session_id}`)
    }
  }

  return (
    <div className="min-h-screen bg-dance-bg p-6">
      <div className="max-w-2xl mx-auto">
        <button onClick={() => navigate('/')} className="text-gray-400 hover:text-white mb-6">
          &larr; Back to Library
        </button>

        <h1 className="text-2xl font-bold mb-6">Connect Phone</h1>

        {loading && <p className="text-gray-400">Creating session...</p>}
        {error && <p className="text-red-400">Error: {error}</p>}

        {session && (
          <div className="bg-dance-card rounded-xl p-6 space-y-6">
            <div className="text-center">
              <h2 className="text-lg text-gray-400 mb-4">Scan QR Code with your phone</h2>
              <div className="inline-block bg-white p-4 rounded-xl">
                <QRCodeSVG value={session.connect_url} size={256} />
              </div>
              <p className="mt-3 text-sm text-gray-500 break-all">{session.connect_url}</p>
            </div>

            <div className="text-center">
              <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full ${phoneConnected ? 'bg-green-900/50 text-green-400' : 'bg-yellow-900/50 text-yellow-400'}`}>
                <span className={`w-3 h-3 rounded-full ${phoneConnected ? 'bg-green-400' : 'bg-yellow-400 animate-pulse'}`} />
                {phoneConnected ? 'Phone Connected' : 'Waiting for phone...'}
              </div>
            </div>

            <div className="text-center">
              <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full ${calibrated ? 'bg-green-900/50 text-green-400' : 'bg-blue-900/50 text-blue-300'}`}>
                <span className={`w-3 h-3 rounded-full ${calibrated ? 'bg-green-400' : 'bg-blue-400 animate-pulse'}`} />
                {calibrated ? 'Calibration Complete' : `Calibrating (${sessionState})`}
              </div>
            </div>

            <button
              onClick={handleStart}
              disabled={!phoneConnected || !calibrated}
              className="w-full py-3 bg-dance-accent rounded-lg hover:bg-purple-700 transition font-semibold text-lg disabled:bg-gray-700 disabled:cursor-not-allowed"
            >
              {phoneConnected && calibrated ? 'Start Game' : 'Waiting for calibration...'}
            </button>

            <div className="text-sm text-gray-500">
              <p>Instructions:</p>
              <ol className="list-decimal list-inside space-y-1 mt-2">
                <li>Open your phone camera and scan the QR code</li>
                <li>Open the link in your phone browser</li>
                <li>Allow camera access</li>
                <li>Stand in front of the camera so your full body is visible</li>
                <li>Wait for calibration to complete</li>
                <li>Press "Start Game" when ready</li>
              </ol>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
