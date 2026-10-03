import React, { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { apiGet, GameResults } from '../api'

export default function Results() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const navigate = useNavigate()
  const [results, setResults] = useState<GameResults | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!sessionId) return
    apiGet(`/api/results/${sessionId}`)
      .then(data => { setResults(data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [sessionId])

  const gradeColor = (g: string) => {
    switch (g) {
      case 'perfect': return 'grade-perfect'
      case 'super': return 'grade-super'
      case 'good': return 'grade-good'
      case 'ok': return 'grade-ok'
      case 'x': return 'grade-x'
      default: return ''
    }
  }

  if (loading) return <div className="min-h-screen bg-dance-bg flex items-center justify-center text-gray-400">Loading results...</div>
  if (!results) return <div className="min-h-screen bg-dance-bg flex items-center justify-center text-gray-400">No results found</div>

  return (
    <div className="min-h-screen bg-dance-bg p-6">
      <div className="max-w-2xl mx-auto text-center">
        <h1 className="text-4xl font-bold mb-6">Dance Results</h1>
        <div className="bg-dance-card rounded-xl p-8 space-y-6">
          <div className="text-6xl font-black text-dance-perfect">{results.total_score.toLocaleString()}</div>
          <div className="text-xl text-gray-400">Total Score</div>

          <div className="grid grid-cols-3 gap-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <div className="text-2xl font-bold">{results.max_combo}x</div>
              <div className="text-sm text-gray-400">Max Combo</div>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-4">
              <div className="text-2xl font-bold">{Math.round(results.accuracy_arms * 100)}%</div>
              <div className="text-sm text-gray-400">Arms</div>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-4">
              <div className="text-2xl font-bold">{Math.round(results.accuracy_legs * 100)}%</div>
              <div className="text-sm text-gray-400">Legs</div>
            </div>
          </div>

          <div className="space-y-2 mt-4">
            {['perfect', 'super', 'good', 'ok', 'x'].map(g => (
              <div key={g} className="flex justify-between items-center px-4 py-2 bg-gray-800/30 rounded">
                <span className={`${gradeColor(g)} capitalize font-semibold`}>{g}</span>
                <span className="text-lg">{results.grade_counts?.[g] || 0}</span>
              </div>
            ))}
          </div>

          <button
            onClick={() => navigate('/')}
            className="w-full py-3 bg-dance-accent rounded-lg hover:bg-purple-700 transition font-semibold text-lg mt-4"
          >
            Back to Library
          </button>
        </div>
      </div>
    </div>
  )
}
