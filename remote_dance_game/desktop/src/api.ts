const envApiBase = (import.meta as any).env?.VITE_API_BASE as string | undefined;

function defaultApiBase(): string {
  if (typeof window === 'undefined') return 'http://localhost:8000';
  const devPorts = new Set(['5173', '5174']);
  if (!devPorts.has(window.location.port)) return window.location.origin;
  const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
  const hostname = window.location.hostname || 'localhost';
  return `${protocol}//${hostname}:8000`;
}

const API_BASE = (envApiBase || defaultApiBase()).replace(/\/$/, '');

async function parseApiError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (data?.detail) return String(data.detail);
  } catch (_) {}
  return `HTTP ${res.status}`;
}

export async function apiGet(path: string) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

export async function apiPost(path: string, body: any) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

export async function apiDelete(path: string) {
  const res = await fetch(`${API_BASE}${path}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

export async function apiUploadVideo(file: File, onProgress?: (fraction: number) => void) {
  return new Promise<any>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}/api/uploads/video`);
    xhr.responseType = 'json';
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response);
      else reject(new Error(String(xhr.response?.detail || `HTTP ${xhr.status}`)));
    };
    xhr.onerror = () => reject(new Error('Upload failed'));
    const form = new FormData();
    form.append('file', file);
    xhr.send(form);
  });
}

export function wsUrl(path: string) {
  const base = new URL(API_BASE);
  const proto = base.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${base.host}${path}`;
}

export function videoUrl(danceId: string) { return `${API_BASE}/video/${danceId}`; }
export function audioUrl(danceId: string) { return `${API_BASE}/audio/${danceId}`; }
export function posterUrl(danceId: string) { return `${API_BASE}/api/dances/${danceId}/poster`; }

export interface DanceTheme {
  name?: string;
  primary?: number[];
  secondary?: number[];
  accent?: number[];
  deep?: number[];
  motif?: string;
}

export interface Dance {
  dance_id: string;
  title: string;
  duration_ms: number;
  difficulty: string;
  created_at: string;
  has_video: boolean;
  has_poster?: boolean;
  preview_mode?: string;
  theme?: DanceTheme;
}

export interface DanceDetail {
  dance_id: string;
  title: string;
  version: number;
  duration_ms: number;
  skeleton_format: string;
  preview_mode: string;
  difficulty: string;
  mirror_mode: boolean;
  created_at: string;
  num_frames: number;
  num_events: number;
  video_path?: string;
  audio_path?: string;
  has_poster?: boolean;
  theme?: DanceTheme;
}

export interface JobStatus {
  job_id: string;
  status: string;
  progress: number;
  stage: string;
  error?: string;
  dance_id?: string;
}

export interface GameSession { session_id: string; dance_id: string; connect_url: string; qr_data: string; }

export interface ScoreEvent {
  type: string;
  timestamp_ms: number;
  grade: 'perfect' | 'super' | 'good' | 'ok' | 'x';
  score: number;
  total_score: number;
  combo: number;
  similarity: number;
  hold_state?: string | null;
  limb_scores?: Record<string, number>;
  timing_offset_ms?: number | null;
  tracking_lost?: boolean;
  pose_age_ms?: number | null;
  coach_pose?: Array<{x: number; y: number; z?: number; v?: number}>;
}

export interface GameResults {
  total_score: number;
  max_combo: number;
  grade_counts: Record<string, number>;
  hold_results: any[];
  accuracy_arms: number;
  accuracy_legs: number;
  accuracy_torso: number;
  timeline_scores: any[];
}
