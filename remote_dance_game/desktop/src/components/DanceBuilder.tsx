import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiGet, apiPost, apiUploadVideo, JobStatus } from '../api'

type SourceMode = 'file' | 'url'

const FRIENDLY_STAGE: Record<string, string> = {
  starting: 'Getting ready',
  resolving_source: 'Getting your video',
  downloading_source: 'Getting your video',
  validating: 'Checking the clip',
  normalizing_video: 'Preparing the footage',
  extracting_audio: 'Listening to the track',
  pose_extraction: 'Learning the choreography',
  processing_choreography: 'Understanding the moves',
  extracting_timing: 'Finding the beat',
  generating_weights: 'Balancing the moves',
  detecting_events: 'Finding special moments',
  rendering_game_video: 'Painting the game stage',
  packaging: 'Finishing your dance',
  done: 'Ready to play',
}

export default function DanceBuilder() {
  const [mode, setMode] = useState<SourceMode>('file')
  const [file, setFile] = useState<File | null>(null)
  const [sourceUrl, setSourceUrl] = useState('')
  const [title, setTitle] = useState('')
  const [building, setBuilding] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [jobId, setJobId] = useState('')
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const navigate = useNavigate()
  const pollRef = useRef<any>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)

  const localPreview = useMemo(() => file ? URL.createObjectURL(file) : '', [file])
  useEffect(() => () => { if (localPreview) URL.revokeObjectURL(localPreview) }, [localPreview])

  const startBuild = async () => {
    try {
      setBuilding(true)
      setJobStatus(null)
      setUploadProgress(0)
      let filePath: string | undefined
      if (mode === 'file') {
        if (!file) throw new Error('Choose a video first')
        const uploaded = await apiUploadVideo(file, setUploadProgress)
        filePath = uploaded.file_path
      } else if (!sourceUrl.trim()) {
        throw new Error('Paste a video link first')
      }

      const result = await apiPost('/api/dances', {
        source_type: mode,
        file_path: filePath,
        source_url: mode === 'url' ? sourceUrl.trim() : undefined,
        clip_start_sec: 0,
        clip_end_sec: null,
        mirror_mode: true,
        title: title || undefined,
        difficulty: 'medium',
      })
      setJobId(result.job_id)
    } catch (e: any) {
      alert(e.message || 'Could not build this dance')
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
          if (status.status === 'completed' && status.dance_id) navigate(`/?focus=${status.dance_id}`)
          if (status.status === 'failed') alert(status.error || 'Build failed')
        }
      } catch (_) {}
    }, 800)
    return () => clearInterval(pollRef.current)
  }, [jobId, navigate])

  const progress = jobStatus?.progress ?? (building && mode === 'file' ? Math.round(uploadProgress * 8) : 0)
  const stage = FRIENDLY_STAGE[jobStatus?.stage || (building ? 'starting' : '')] || 'Building your dance'

  if (building) {
    return (
      <main className="builder-processing">
        {localPreview && <video className="builder-video-bg" src={localPreview} autoPlay loop muted playsInline />}
        <div className="builder-processing-wash" />
        <button className="builder-back minimal" onClick={() => navigate('/')}>←</button>
        <div className="processing-center">
          <div className="processing-ring" style={{ '--progress': `${progress * 3.6}deg` } as React.CSSProperties}>
            <div><strong>{progress}%</strong><span>{stage}</span></div>
          </div>
          <h1>Turning footage into a game.</h1>
          <p>The choreography, beat, bright coach and stage are generated once. Take your time — playback stays fast later.</p>
        </div>
        <div className="processing-steps">
          <span className={progress > 18 ? 'done' : ''}>VIDEO</span>
          <i />
          <span className={progress > 63 ? 'done' : ''}>MOVES</span>
          <i />
          <span className={progress > 83 ? 'done' : ''}>STYLE</span>
          <i />
          <span className={progress > 98 ? 'done' : ''}>READY</span>
        </div>
      </main>
    )
  }

  return (
    <main className="builder-screen">
      <div className="builder-aurora" />
      <header className="builder-header">
        <button className="builder-back" onClick={() => navigate('/')}>←</button>
        <div className="tv-logo">DANCE<span>FLOW</span></div>
        <div className="source-tabs">
          <button className={mode === 'file' ? 'active' : ''} onClick={() => setMode('file')}>FILE</button>
          <button className={mode === 'url' ? 'active' : ''} onClick={() => setMode('url')}>LINK</button>
        </div>
      </header>

      <section className="builder-main">
        <div className="builder-copy">
          <div className="active-eyebrow">CREATE A DANCE</div>
          <h1>Drop in a video.<br/>Get a playable stage.</h1>
          <p>One visible main dancer works best. Everything else — movement map, timing, colors and effects — is automatic.</p>
        </div>

        {mode === 'file' ? (
          <button className={`video-drop ${file ? 'has-file' : ''}`} onClick={() => inputRef.current?.click()}>
            <input ref={inputRef} type="file" accept="video/*,.mkv,.webm" hidden onChange={e => setFile(e.target.files?.[0] || null)} />
            {localPreview && <video src={localPreview} muted autoPlay loop playsInline />}
            <div className="drop-wash" />
            <div className="drop-content">
              <div className="drop-plus">＋</div>
              <strong>{file ? file.name : 'CHOOSE VIDEO'}</strong>
              <span>{file ? `${(file.size / 1024 / 1024).toFixed(0)} MB · tap to replace` : 'MP4 · MOV · MKV · WEBM'}</span>
            </div>
          </button>
        ) : (
          <div className="link-panel">
            <span>VIDEO LINK</span>
            <input autoFocus value={sourceUrl} onChange={e => setSourceUrl(e.target.value)} placeholder="Paste YouTube, TikTok or another supported link" />
          </div>
        )}

        <div className="builder-bottom-row">
          <input className="title-ghost" value={title} onChange={e => setTitle(e.target.value)} placeholder="Optional title" />
          <button className="tv-play build-action" disabled={mode === 'file' ? !file : !sourceUrl.trim()} onClick={startBuild}>GENERATE DANCE →</button>
        </div>
      </section>
    </main>
  )
}
