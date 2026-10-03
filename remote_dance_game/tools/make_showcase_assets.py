from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ["DANCE_DISABLE_YOLO"] = "1"
from services.stylized_video import render_game_video

OUT = ROOT / "desktop" / "public" / "showcase"
OUT.mkdir(parents=True, exist_ok=True)
source = OUT / "source.mp4"
audio = OUT / "audio.m4a"
game = OUT / "game.mp4"
poster = OUT / "poster.jpg"

W,H,FPS,DUR = 640,360,18,8
writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W,H))
frames=[]
for i in range(FPS*DUR):
    t=i/FPS
    img=np.zeros((H,W,3),np.uint8)
    # deliberately ordinary source room, so transformation is visible in recording
    img[:] = (42,45,48)
    cv2.rectangle(img,(0,int(H*.76)),(W,H),(55,55,58),-1)
    cv2.rectangle(img,(80,85),(270,390),(34,38,42),-1)
    cv2.circle(img,(790,135),62,(58,68,78),-1)
    cx=int(W*(.5+.11*math.sin(t*1.55))); cy=int(H*.48)
    pose=[{'x':0.,'y':0.,'z':0.,'v':0.} for _ in range(33)]
    def p(k,x,y): pose[k]={'x':x/W,'y':y/H,'z':0.,'v':1.}
    p(0,cx,cy-126); p(11,cx-46,cy-68);p(12,cx+46,cy-68)
    wave=math.sin(t*3.2)
    p(13,cx-88,cy-32-int(wave*70));p(15,cx-135,cy-6-int(wave*105))
    p(14,cx+88,cy-32+int(wave*70));p(16,cx+135,cy-6+int(wave*105))
    p(23,cx-32,cy+35);p(24,cx+32,cy+35)
    kick=abs(math.sin(t*2.25))*40
    p(25,cx-45,cy+116);p(27,cx-62-int(kick),cy+202-int(kick*.2))
    p(26,cx+45,cy+116);p(28,cx+62+int(kick),cy+202-int(kick*.2))
    color=(88,128,218)
    for a,b in ((11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)):
        pa,pb=pose[a],pose[b]
        cv2.line(img,(int(pa['x']*W),int(pa['y']*H)),(int(pb['x']*W),int(pb['y']*H)),color,32,cv2.LINE_AA)
    cv2.circle(img,(cx,cy-126),34,(173,185,200),-1,cv2.LINE_AA)
    writer.write(img)
    frames.append({'t_ms':int(t*1000),'landmarks':pose,'world_landmarks':pose})
writer.release()
subprocess.run(['ffmpeg','-y','-f','lavfi','-i','sine=frequency=246:sample_rate=44100','-f','lavfi','-i','sine=frequency=492:sample_rate=44100','-filter_complex','[0:a]volume=0.30[a0];[1:a]volume=0.12[a1];[a0][a1]amix=inputs=2','-t',str(DUR),'-c:a','aac',str(audio)],check=True)
beats=list(range(0,DUR*1000,500)); strong=list(range(0,DUR*1000,1000))
render_game_video(str(source),str(game),{'fps':FPS,'duration_ms':DUR*1000,'total_frames':len(frames),'frames':frames},{'tempo':120,'beat_ms':beats,'strong_beat_ms':strong},str(audio),'Night Move',str(poster),False,W,H,FPS)
print(game)
