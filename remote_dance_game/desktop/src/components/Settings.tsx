import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiGet, apiPost } from '../api'

export default function Settings() {
  const navigate = useNavigate()
  const [thresholds, setThresholds] = useState({ perfect: 0.92, super: 0.82, good: 0.68, ok: 0.50 })
  const [sensitivity, setSensitivity] = useState(1.0)
  const [matchingWindowMs, setMatchingWindowMs] = useState(120)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    apiGet('/api/settings').then(data => {
      if (data.scoring_thresholds_base) setThresholds(data.scoring_thresholds_base)
      if (typeof data.sensitivity === 'number') setSensitivity(data.sensitivity)
      if (typeof data.matching_window_ms === 'number') setMatchingWindowMs(data.matching_window_ms)
    }).catch(() => {})
  }, [])

  const handleSave = async () => {
    try {
      await apiPost('/api/settings', {
        scoring_thresholds: thresholds,
        sensitivity,
        matching_window_ms: matchingWindowMs,
      })
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch (e: any) {
      alert('Save failed: ' + e.message)
    }
  }

  return (
    <div className="min-h-screen bg-dance-bg p-6">
      <div className="max-w-2xl mx-auto">
        <button onClick={() => navigate('/')} className="text-gray-400 hover:text-white mb-6">
          &larr; Back to Library
        </button>

        <h1 className="text-2xl font-bold mb-6">Settings</h1>

        <div className="bg-dance-card rounded-xl p-6 space-y-6">
          <div>
            <h3 className="text-lg font-semibold mb-3">Scoring Thresholds</h3>
            {Object.entries(thresholds).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between py-2">
                <span className="capitalize w-24">{key}</span>
                <input
                  type="range"
                  min={0.3}
                  max={1.0}
                  step={0.01}
                  value={value}
                  onChange={e => setThresholds({ ...thresholds, [key]: parseFloat(e.target.value) })}
                  className="flex-1 mx-4 accent-dance-accent"
                />
                <span className="w-12 text-right">{value.toFixed(2)}</span>
              </div>
            ))}
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-3">Sensitivity Multiplier</h3>
            <div className="flex items-center justify-between py-2">
              <span>Easier</span>
              <input
                type="range"
                min={0.5}
                max={1.5}
                step={0.1}
                value={sensitivity}
                onChange={e => setSensitivity(parseFloat(e.target.value))}
                className="flex-1 mx-4 accent-dance-accent"
              />
              <span className="w-12 text-right">{sensitivity.toFixed(1)}</span>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-3">Timing Matching Window</h3>
            <div className="flex items-center justify-between py-2">
              <span>Tight</span>
              <input
                type="range"
                min={60}
                max={250}
                step={5}
                value={matchingWindowMs}
                onChange={e => setMatchingWindowMs(parseInt(e.target.value, 10))}
                className="flex-1 mx-4 accent-dance-accent"
              />
              <span className="w-16 text-right">{matchingWindowMs}ms</span>
            </div>
          </div>

          <button
            onClick={handleSave}
            className="w-full py-3 bg-dance-accent rounded-lg hover:bg-purple-700 transition font-semibold"
          >
            {saved ? 'Saved!' : 'Save Settings'}
          </button>
        </div>
      </div>
    </div>
  )
}
