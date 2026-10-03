import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiGet, apiDelete, Dance } from '../api'

export default function Library() {
  const [dances, setDances] = useState<Dance[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  const loadDances = async () => {
    try {
      setLoading(true)
      const data = await apiGet('/api/dances')
      setDances(data)
      setError('')
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadDances() }, [])

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this dance?')) return
    try {
      await apiDelete(`/api/dances/${id}`)
      loadDances()
    } catch (e: any) {
      alert('Delete failed: ' + e.message)
    }
  }

  const handlePlay = async (danceId: string) => {
    navigate(`/connect/${danceId}`)
  }

  const formatDuration = (ms: number) => {
    const s = Math.floor(ms / 1000)
    const m = Math.floor(s / 60)
    return `${m}:${(s % 60).toString().padStart(2, '0')}`
  }

  return (
    <div className="min-h-screen bg-dance-bg p-6">
      <div className="max-w-5xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold">Dance Coach Game</h1>
          <div className="flex gap-3">
            <button
              onClick={() => navigate('/settings')}
              className="px-4 py-2 bg-dance-card rounded-lg hover:bg-gray-700 transition"
            >
              Settings
            </button>
            <button
              onClick={() => navigate('/build')}
              className="px-4 py-2 bg-dance-accent rounded-lg hover:bg-purple-700 transition font-semibold"
            >
              + Create Dance
            </button>
          </div>
        </div>

        {loading && <p className="text-gray-400">Loading dances...</p>}
        {error && <p className="text-red-400">Error: {error}</p>}

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {dances.map(d => (
            <div key={d.dance_id} className="bg-dance-card rounded-xl p-5 border border-gray-800 hover:border-dance-accent transition">
              <h3 className="text-lg font-semibold mb-2 truncate">{d.title}</h3>
              <div className="flex gap-4 text-sm text-gray-400 mb-4">
                <span>{formatDuration(d.duration_ms)}</span>
                <span className="capitalize">{d.difficulty}</span>
                <span>{d.has_video ? 'Video' : 'No video'}</span>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => handlePlay(d.dance_id)}
                  className="flex-1 px-4 py-2 bg-dance-accent rounded-lg hover:bg-purple-700 transition font-semibold"
                >
                  Play
                </button>
                <button
                  onClick={() => handleDelete(d.dance_id)}
                  className="px-3 py-2 bg-red-900/50 rounded-lg hover:bg-red-800 transition text-red-300"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>

        {dances.length === 0 && !loading && (
          <div className="text-center py-20 text-gray-500">
            <p className="text-xl mb-4">No dances yet</p>
            <button
              onClick={() => navigate('/build')}
              className="px-6 py-3 bg-dance-accent rounded-lg hover:bg-purple-700 transition font-semibold text-lg"
            >
              Create Your First Dance
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
