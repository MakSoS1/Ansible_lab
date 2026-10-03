import React, { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiGet, apiPost, apiUploadVideo, JobStatus } from '../api'

type SourceMode = 'file' | 'url'

export default function DanceBuilder() {
  const [mode, setMode] = useState<SourceMode>('file')
  const [file, setFile] = useState<File | null>(null)
  const [sourceUrl, setSourceUrl] = useState('')
  const [title, setTitle] = useState('')
  const [clipStart, setClipStart] = useState(0)
  const [clipEnd, setClipEnd] = useState(0)
  const [mirrorMode, setMirrorMode] = useState(true)
  const [difficulty, setDifficulty] = useState('medium')
  const [building, setBuilding] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [jobId, setJobId] = useState('')
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const navigate = useNavigate()
  const pollRef = useRef<any>(null)

  const startBuild = async () => {
    try {
      setBuilding(true)
      setJobStatus(null)
      setUploadProgress(0)

      let filePath: string | undefined
      if (mode === 'file') {
        if (!file) throw new Error('Choose a video file first')
        const uploaded = await apiUploadVideo(file, setUploadProgress)
        filePath = uploaded.file_path
      } else if (!sourceUrl.trim()) {
        throw new Error('Paste a YouTube, TikTok or other supported video URL')
      }

      const result = await apiPost('/api/dances', {
        source_type: mode,
        file_path: filePath,
        source_url: mode === 'url' ? sourceUrl.trim() : undefined,
        clip_start_sec: clipStart,
        clip_end_sec: clipEnd > 0 ? clipEnd : null,
        mirror_mode: mirrorMode,
        title: title || undefined,
        difficulty,
      })
      setJobId(result.job_id)
    } catch (e: any) {
      alert('Build failed: ' + e.message)
      setBuilding(false)
    }
  }

  useEffect(() => {
    if (!jobId) return
    pollRef.current = setInterval(async () => {
      try {
        const status = await apiGet(`/api/jobs/${jobId}`)
        setJobStatus(status)
        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(pollRef.current)
          setBuilding(false)
          if (status.status === 'completed' && status.dance_id) navigate(`/connect/${status.dance_id}`)
          if (status.status === 'failed') alert('Build failed: ' + status.error)
        }
      } catch (_) {}
    }, 800)
    return () => clearInterval(pollRef.current)
  }, [jobId, navigate])

  const progress = jobStatus?.progress || 0

  return (
    <div className="min-h-screen dance-shell p-6 overflow-auto">
      <div className="max-w-3xl mx-auto pb-12">
        <button onClick={() => navigate('/')} className="text-white/60 hover:text-white mb-6">← Library</button>
        <div className="mb-7">
          <p className="dance-kicker">CHOREOGRAPHY LAB</p>
          <h1 className="text-4xl font-black tracking-tight">Turn any dance video into a playable routine.</h1>
          <p className="text-white/55 mt-2">The game extracts pose, rhythm and movement weights once. Gameplay stays local and low-latency.</p>
        </div>

        <div className="glass-card rounded-3xl p-6 md:p-8 space-y-6">
          <div className="grid grid-cols-2 bg-black/25 p-1 rounded-2xl">
            <button onClick={() => setMode('file')} className={`py-3 rounded-xl font-bold transition ${mode === 'file' ? 'bg-white text-black' : 'text-white/60'}`}>Upload file</button>
            <button onClick={() => setMode('url')} className={`py-3 rounded-xl font-bold transition ${mode === 'url' ? 'bg-white text-black' : 'text-white/60'}`}>Paste link</button>
          </div>

          {mode === 'file' ? (
            <label className="upload-drop block cursor-pointer rounded-3xl p-8 text-center">
              <input type="file" accept="video/*,.mkv,.webm" className="hidden" onChange={e => setFile(e.target.files?.[0] || null)} />
              <div className="text-5xl mb-3">✦</div>
              <div className="font-black text-xl">{file ? file.name : 'Choose a dance video'}</div>
              <div className="text-white/45 text-sm mt-2">MP4, MOV, MKV, WebM — up to 2 GiB</div>
              {file && <div className="text-white/60 text-sm mt-3">{(file.size / 1024 / 1024).toFixed(1)} MB</div>}
            </label>
          ) : (
            <div>
              <label className="block text-sm text-white/50 mb-2">Video URL</label>
              <input value={sourceUrl} onChange={e => setSourceUrl(e.target.value)} className="dance-input" placeholder="https://www.youtube.com/... or https://www.tiktok.com/..." />
              <p className="text-xs text-white/35 mt-2">Use videos you have permission to process. Private/login-only pages may require provider cookies and are intentionally not handled here.</p>
            </div>
          )}

          <div>
            <label className="block text-sm text-white/50 mb-2">Routine title</label>
            <input value={title} onChange={e => setTitle(e.target.value)} className="dance-input" placeholder="Auto-detected if left empty" />
          </div>

          <div className="grid md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm text-white/50 mb-2">Start, sec</label>
              <input type="number" value={clipStart} onChange={e => setClipStart(parseFloat(e.target.value) || 0)} className="dance-input" min={0} />
            </div>
            <div>
              <label className="block text-sm text-white/50 mb-2">End, sec</label>
              <input type="number" value={clipEnd} onChange={e => setClipEnd(parseFloat(e.target.value) || 0)} className="dance-input" min={0} />
            </div>
            <div>
              <label className="block text-sm text-white/50 mb-2">Difficulty</label>
              <select value={difficulty} onChange={e => setDifficulty(e.target.value)} className="dance-input">
                <option value="easy">Easy</option><option value="medium">Medium</option><option value="hard">Hard</option>
              </select>
            </div>
          </div>

          <label className="flex items-center gap-3 cursor-pointer rounded-2xl bg-white/5 p-4">
            <input type="checkbox" checked={mirrorMode} onChange={e => setMirrorMode(e.target.checked)} className="w-5 h-5 accent-fuchsia-500" />
            <div><div className="font-bold">Mirror coach</div><div className="text-xs text-white/45">Natural “follow the person on screen” mode.</div></div>
          </label>

          {building && (
            <div className="rounded-2xl bg-black/25 p-4 space-y-2">
              <div className="flex justify-between text-sm"><span className="text-white/60">{jobStatus?.stage || (mode === 'file' && uploadProgress < 1 ? 'uploading' : 'starting')}</span><span>{jobStatus ? `${progress}%` : `${Math.round(uploadProgress * 100)}%`}</span></div>
              <div className="h-3 rounded-full bg-white/10 overflow-hidden"><div className="h-full dance-progress" style={{ width: `${jobStatus ? progress : uploadProgress * 100}%` }} /></div>
              <div className="text-xs text-white/35">Pose extraction is the expensive one-time step; it never runs during gameplay.</div>
            </div>
          )}

          <button onClick={startBuild} disabled={building || (mode === 'file' ? !file : !sourceUrl.trim())} className="dance-primary w-full py-4 rounded-2xl font-black text-lg disabled:opacity-35 disabled:cursor-not-allowed">
            {building ? 'Building routine…' : 'Generate playable dance'}
          </button>
        </div>
      </div>
    </div>
  )
}
