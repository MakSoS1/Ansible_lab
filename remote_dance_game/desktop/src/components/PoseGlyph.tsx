import React from 'react'

type Point = { x: number; y: number; v?: number }

type Props = {
  pose?: Point[]
  active?: boolean
  variant?: number
}

const CONNECTIONS: Array<[number, number]> = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
  [24, 26], [26, 28], [27, 31], [28, 32],
]

const FALLBACKS: Point[][] = [
  [
    {x:.50,y:.15,v:1},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},
    {x:.43,y:.29,v:1},{x:.57,y:.29,v:1},{x:.35,y:.43,v:1},{x:.69,y:.20,v:1},{x:.29,y:.56,v:1},{x:.80,y:.08,v:1},
    ...Array(6).fill({x:0,y:0,v:0}),
    {x:.45,y:.54,v:1},{x:.56,y:.54,v:1},{x:.42,y:.72,v:1},{x:.60,y:.72,v:1},{x:.39,y:.91,v:1},{x:.66,y:.90,v:1},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:.36,y:.96,v:1},{x:.70,y:.96,v:1},
  ],
  [
    {x:.50,y:.14,v:1},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},
    {x:.43,y:.29,v:1},{x:.57,y:.29,v:1},{x:.30,y:.40,v:1},{x:.72,y:.42,v:1},{x:.18,y:.37,v:1},{x:.84,y:.50,v:1},
    ...Array(6).fill({x:0,y:0,v:0}),
    {x:.45,y:.54,v:1},{x:.56,y:.54,v:1},{x:.38,y:.71,v:1},{x:.63,y:.70,v:1},{x:.30,y:.88,v:1},{x:.73,y:.88,v:1},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:.26,y:.93,v:1},{x:.78,y:.93,v:1},
  ],
  [
    {x:.50,y:.14,v:1},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},
    {x:.43,y:.29,v:1},{x:.57,y:.29,v:1},{x:.38,y:.43,v:1},{x:.65,y:.43,v:1},{x:.31,y:.58,v:1},{x:.80,y:.43,v:1},
    ...Array(6).fill({x:0,y:0,v:0}),
    {x:.45,y:.54,v:1},{x:.56,y:.54,v:1},{x:.42,y:.72,v:1},{x:.61,y:.68,v:1},{x:.41,y:.90,v:1},{x:.76,y:.77,v:1},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:0,y:0,v:0},{x:.38,y:.95,v:1},{x:.84,y:.80,v:1},
  ],
]

function normalizePose(raw?: Point[], variant = 0) {
  const pose = raw && raw.length >= 29 ? raw : FALLBACKS[variant % FALLBACKS.length]
  const used = [0, 11,12,13,14,15,16,23,24,25,26,27,28,31,32]
  const visible = used
    .map(i => ({i, p: pose[i]}))
    .filter(x => x.p && (x.p.v ?? 1) > .18 && Number.isFinite(x.p.x) && Number.isFinite(x.p.y))
  if (!visible.length) return { pose, map: (p: Point) => ({x:p.x*100,y:p.y*100}) }
  const xs = visible.map(x => x.p.x)
  const ys = visible.map(x => x.p.y)
  const minX = Math.min(...xs), maxX = Math.max(...xs)
  const minY = Math.min(...ys), maxY = Math.max(...ys)
  const spanX = Math.max(.08, maxX - minX)
  const spanY = Math.max(.12, maxY - minY)
  const scale = Math.min(68 / spanX, 76 / spanY)
  const cx = (minX + maxX) / 2
  const cy = (minY + maxY) / 2
  return {
    pose,
    map: (p: Point) => ({
      x: 50 + (p.x - cx) * scale,
      y: 51 + (p.y - cy) * scale,
    }),
  }
}

export default function PoseGlyph({ pose: rawPose, active = false, variant = 0 }: Props) {
  const { pose, map } = normalizePose(rawPose, variant)
  const head = pose[0]
  const hp = head && (head.v ?? 1) > .18 ? map(head) : {x:50,y:18}
  return (
    <svg viewBox="0 0 100 100" className={`pose-glyph ${active ? 'active' : ''}`} aria-hidden="true">
      <g className="pose-glyph-glow">
        {CONNECTIONS.map(([a,b]) => {
          const pa = pose[a], pb = pose[b]
          if (!pa || !pb || (pa.v ?? 1) <= .18 || (pb.v ?? 1) <= .18) return null
          const A = map(pa), B = map(pb)
          return <line key={`${a}-${b}`} x1={A.x} y1={A.y} x2={B.x} y2={B.y} />
        })}
        <circle cx={hp.x} cy={hp.y} r="7.2" />
      </g>
    </svg>
  )
}
