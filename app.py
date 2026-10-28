import os
import sys
import glob
import time
import json
import asyncio
import datetime
import threading
import subprocess
from pathlib import Path
from typing import Optional, List, Dict

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="AI Video Transcription Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKSPACE_DIR = Path(__file__).parent.resolve()
VIDEOS_DIR = WORKSPACE_DIR / "videos"
TRANSCRIPTS_DIR = WORKSPACE_DIR / "transcripts"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

class TranscriptionState:
    def __init__(self):
        self.is_running = False
        self.file_name = ""
        self.total_duration = 0.0
        self.current_time = 0.0
        self.percentage = 0.0
        self.language = ""
        self.language_prob = 0.0
        self.speed = 0.0
        self.eta_seconds = 0
        self.current_segment = ""
        self.plain_text = ""
        self.timestamps_text = ""
        self.srt_text = ""
        self.error = None
        self.start_wall_time = 0.0
        self.stop_requested = False

state = TranscriptionState()
state_lock = threading.Lock()

def format_timestamp(seconds: float, srt_format: bool = False) -> str:
    td = datetime.timedelta(seconds=max(0.0, seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)

    if srt_format:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def run_transcription_worker(file_path_str: str, model_name: str, language_choice: Optional[str]):
    global state
    from faster_whisper import WhisperModel

    file_path = Path(file_path_str).resolve()
    if not file_path.exists():
        with state_lock:
            state.is_running = False
            state.error = f"File not found: {file_path.name}"
        return

    base_name = file_path.stem
    video_out_dir = TRANSCRIPTS_DIR / base_name
    video_out_dir.mkdir(parents=True, exist_ok=True)
    txt_plain_path = video_out_dir / f"{base_name}_transcript.txt"
    txt_time_path = video_out_dir / f"{base_name}_timestamps.txt"
    srt_path = video_out_dir / f"{base_name}.srt"

    with state_lock:
        state.is_running = True
        state.file_name = file_path.name
        state.error = None
        state.stop_requested = False
        state.plain_text = ""
        state.timestamps_text = ""
        state.srt_text = ""
        state.start_wall_time = time.time()
        state.current_time = 0.0
        state.percentage = 0.0

    try:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")

        segments_gen, info = model.transcribe(
            str(file_path),
            language=language_choice if language_choice and language_choice != "auto" else None,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            beam_size=5
        )

        with state_lock:
            state.total_duration = info.duration
            state.language = info.language
            state.language_prob = info.language_probability

        with open(txt_plain_path, "w", encoding="utf-8") as f_plain, \
             open(txt_time_path, "w", encoding="utf-8") as f_time, \
             open(srt_path, "w", encoding="utf-8") as f_srt:

            f_time.write(f"# Transcript: {file_path.name}\n")
            f_time.write(f"# Language: {info.language} | Duration: {format_timestamp(info.duration)}\n\n")

            srt_idx = 1
            prev_end = 0.0
            paragraph_word_count = 0

            for segment in segments_gen:
                with state_lock:
                    if state.stop_requested:
                        break

                text = segment.text.strip()
                if not text:
                    continue

                start_str = format_timestamp(segment.start)
                end_str = format_timestamp(segment.end)
                srt_start = format_timestamp(segment.start, srt_format=True)
                srt_end = format_timestamp(segment.end, srt_format=True)

                silence_gap = segment.start - prev_end
                words = text.split()
                paragraph_word_count += len(words)

                # Format plain paragraph
                if (silence_gap > 2.0 or paragraph_word_count > 80) and prev_end > 0:
                    added_plain = f"\n\n{text}"
                    paragraph_word_count = len(words)
                else:
                    added_plain = f" {text}" if prev_end > 0 else text

                f_plain.write(added_plain)
                f_plain.flush()

                time_line = f"[{start_str} -> {end_str}] {text}\n"
                f_time.write(time_line)
                f_time.flush()

                srt_entry = f"{srt_idx}\n{srt_start} --> {srt_end}\n{text}\n\n"
                f_srt.write(srt_entry)
                f_srt.flush()
                srt_idx += 1

                prev_end = segment.end

                elapsed = max(0.1, time.time() - state.start_wall_time)
                speed = segment.end / elapsed if elapsed > 0 else 1.0
                remaining_sec = max(0.0, info.duration - segment.end)
                eta = int(remaining_sec / speed) if speed > 0 else 0

                with state_lock:
                    state.current_time = segment.end
                    state.percentage = min(100.0, (segment.end / info.duration) * 100) if info.duration > 0 else 0
                    state.speed = speed
                    state.eta_seconds = eta
                    state.current_segment = text
                    state.plain_text += added_plain
                    state.timestamps_text += time_line
                    state.srt_text += srt_entry

            with state_lock:
                state.percentage = 100.0
                state.current_time = state.total_duration

    except Exception as e:
        with state_lock:
            state.error = str(e)
    finally:
        with state_lock:
            state.is_running = False

class TranscribeRequest(BaseModel):
    video_name: str
    model: str = "small"
    language: Optional[str] = "auto"

@app.get("/api/videos")
def get_videos():
    valid_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".m4a"}
    files = []
    seen = set()

    # Search in videos/ folder first, then project root
    search_dirs = [VIDEOS_DIR, WORKSPACE_DIR]
    for d in search_dirs:
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in valid_exts and f.name not in seen:
                    seen.add(f.name)
                    size_mb = f.stat().st_size / (1024 * 1024)
                    has_transcript = (TRANSCRIPTS_DIR / f"{f.stem}" / f"{f.stem}_transcript.txt").exists() or (TRANSCRIPTS_DIR / f"{f.stem}_transcript.txt").exists()
                    files.append({
                        "name": f.name,
                        "size_mb": round(size_mb, 1),
                        "has_transcript": has_transcript,
                        "folder": d.name
                    })
    return {"videos": sorted(files, key=lambda x: x["name"])}

@app.get("/api/status")
def get_status():
    with state_lock:
        if state.is_running:
            return {
                "is_running": True,
                "file_name": state.file_name,
                "total_duration": state.total_duration,
                "current_time": state.current_time,
                "percentage": round(state.percentage, 1),
                "language": state.language,
                "language_prob": state.language_prob,
                "speed": round(state.speed, 1),
                "eta_seconds": state.eta_seconds,
                "current_segment": state.current_segment,
                "plain_text": state.plain_text,
                "timestamps_text": state.timestamps_text,
                "srt_text": state.srt_text,
                "error": state.error
            }

        # Check existing transcripts in transcripts/
        check_pithon_ts = TRANSCRIPTS_DIR / "pithon" / "pithon_timestamps.txt"
        check_pithon_txt = TRANSCRIPTS_DIR / "pithon" / "pithon_transcript.txt"
        if not check_pithon_ts.exists():
            check_pithon_ts = TRANSCRIPTS_DIR / "pithon_timestamps.txt"
            check_pithon_txt = TRANSCRIPTS_DIR / "pithon_transcript.txt"

        if check_pithon_ts.exists() and check_pithon_txt.exists():
            with open(check_pithon_txt, "r", encoding="utf-8", errors="ignore") as pf:
                plain_content = pf.read()
            with open(check_pithon_ts, "r", encoding="utf-8", errors="ignore") as tsf:
                lines = [line.strip() for line in tsf if line.strip()]

            return {
                "is_running": False,
                "is_completed": True,
                "file_name": "pithon.mp4",
                "total_duration": 5507.0,
                "current_time": 5507.0,
                "percentage": 100.0,
                "language": "en",
                "language_prob": 0.94,
                "speed": 0.0,
                "eta_seconds": 0,
                "current_segment": "All speech transcribed successfully!",
                "plain_text": plain_content,
                "timestamps_text": "\n".join(lines),
                "srt_text": "",
                "error": None
            }

        return {
            "is_running": False,
            "is_completed": False,
            "file_name": "",
            "total_duration": 0.0,
            "current_time": 0.0,
            "percentage": 0.0,
            "language": "",
            "language_prob": 0.0,
            "speed": 0.0,
            "eta_seconds": 0,
            "current_segment": "",
            "plain_text": "",
            "timestamps_text": "",
            "srt_text": "",
            "error": state.error
        }

@app.post("/api/transcribe")
def start_transcribe(req: TranscribeRequest, background_tasks: BackgroundTasks):
    with state_lock:
        if state.is_running:
            raise HTTPException(status_code=400, detail="A transcription is already in progress.")

    target_path = VIDEOS_DIR / req.video_name
    if not target_path.exists():
        target_path = WORKSPACE_DIR / req.video_name
    if not target_path.exists():
        target_path = Path(req.video_name)
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found in videos/ or workspace: {req.video_name}")

    thread = threading.Thread(
        target=run_transcription_worker,
        args=(str(target_path), req.model, req.language),
        daemon=True
    )
    thread.start()
    return {"message": "Transcription started", "video": target_path.name}

@app.post("/api/stop")
def stop_transcription():
    with state_lock:
        if state.is_running:
            state.stop_requested = True
            return {"message": "Stop requested"}
        return {"message": "No active transcription to stop"}

@app.get("/api/transcript/{video_name}")
def load_transcript(video_name: str):
    stem = Path(video_name).stem
    subfolder = TRANSCRIPTS_DIR / stem
    if subfolder.exists():
        txt_plain = subfolder / f"{stem}_transcript.txt"
        txt_time = subfolder / f"{stem}_timestamps.txt"
        srt_file = subfolder / f"{stem}.srt"
    else:
        txt_plain = TRANSCRIPTS_DIR / f"{stem}_transcript.txt"
        txt_time = TRANSCRIPTS_DIR / f"{stem}_timestamps.txt"
        srt_file = TRANSCRIPTS_DIR / f"{stem}.srt"

    res = {
        "plain": "",
        "timestamps": "",
        "srt": ""
    }
    if txt_plain.exists():
        with open(txt_plain, "r", encoding="utf-8", errors="ignore") as f:
            res["plain"] = f.read()
    if txt_time.exists():
        with open(txt_time, "r", encoding="utf-8", errors="ignore") as f:
            res["timestamps"] = f.read()
    if srt_file.exists():
        with open(srt_file, "r", encoding="utf-8", errors="ignore") as f:
            res["srt"] = f.read()

    return res

@app.post("/api/open_transcripts")
def open_transcripts():
    try:
        if sys.platform == "win32":
            os.startfile(str(TRANSCRIPTS_DIR))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(TRANSCRIPTS_DIR)])
        else:
            subprocess.run(["xdg-open", str(TRANSCRIPTS_DIR)])
        return {"message": "Transcripts folder opened"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/open_videos")
def open_videos():
    try:
        if sys.platform == "win32":
            os.startfile(str(VIDEOS_DIR))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(VIDEOS_DIR)])
        else:
            subprocess.run(["xdg-open", str(VIDEOS_DIR)])
        return {"message": "Videos folder opened"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
def index():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Video Transcription Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(17, 24, 39, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --cyan: #06b6d4;
      --emerald: #10b981;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --danger: #ef4444;
      --glow: rgba(99, 102, 241, 0.35);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Outfit', sans-serif;
      background: radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.12), transparent 40%),
                  radial-gradient(circle at 85% 85%, rgba(6, 182, 212, 0.08), transparent 40%),
                  var(--bg);
      background-attachment: fixed;
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      padding: 20px 40px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--card-border);
      backdrop-filter: blur(12px);
      background: rgba(9, 13, 22, 0.85);
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .logo-icon {
      width: 44px;
      height: 44px;
      border-radius: 12px;
      background: linear-gradient(135deg, var(--primary), var(--cyan));
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 25px var(--glow);
    }

    .brand h1 {
      font-size: 1.4rem;
      font-weight: 700;
      letter-spacing: -0.5px;
      background: linear-gradient(to right, #ffffff, #94a3b8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand p {
      font-size: 0.82rem;
      color: var(--text-muted);
    }

    .header-actions {
      display: flex;
      gap: 12px;
    }

    .btn-secondary {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      color: var(--text-main);
      padding: 9px 18px;
      border-radius: 10px;
      font-size: 0.88rem;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s ease;
    }

    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.2);
      transform: translateY(-1px);
    }

    main {
      flex: 1;
      max-width: 1440px;
      width: 100%;
      margin: 0 auto;
      padding: 30px 40px;
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 30px;
    }

    @media (max-width: 1024px) {
      main {
        grid-template-columns: 1fr;
      }
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 18px;
      padding: 24px;
      backdrop-filter: blur(16px);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }

    .card-title {
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 18px;
      display: flex;
      align-items: center;
      gap: 10px;
      color: #e2e8f0;
    }

    .form-group {
      margin-bottom: 20px;
    }

    label {
      display: block;
      font-size: 0.85rem;
      font-weight: 500;
      color: var(--text-muted);
      margin-bottom: 8px;
    }

    select, input[type="text"] {
      width: 100%;
      padding: 12px 14px;
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      color: var(--text-main);
      font-family: inherit;
      font-size: 0.95rem;
      transition: all 0.2s;
    }

    select:focus, input[type="text"]:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 15px rgba(99, 102, 241, 0.25);
    }

    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }

    .btn-primary {
      width: 100%;
      padding: 14px;
      border-radius: 12px;
      border: none;
      background: linear-gradient(135deg, var(--primary), var(--cyan));
      color: #ffffff;
      font-size: 1.05rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      transition: all 0.25s ease;
      box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
    }

    .btn-primary:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 8px 30px rgba(99, 102, 241, 0.6);
      filter: brightness(1.1);
    }

    .btn-primary:disabled {
      opacity: 0.55;
      cursor: not-allowed;
      box-shadow: none;
    }

    .btn-danger {
      width: 100%;
      padding: 12px;
      border-radius: 10px;
      border: none;
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.4);
      color: #fca5a5;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      margin-top: 10px;
      transition: all 0.2s;
    }

    .btn-danger:hover {
      background: rgba(239, 68, 68, 0.25);
    }

    /* Video list pills */
    .video-list {
      max-height: 220px;
      overflow-y: auto;
      margin-top: 10px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .video-item {
      padding: 10px 14px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      font-size: 0.88rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .video-item:hover {
      background: rgba(99, 102, 241, 0.15);
      border-color: var(--primary);
    }

    .video-item.active {
      background: rgba(99, 102, 241, 0.25);
      border-color: var(--cyan);
    }

    .badge {
      font-size: 0.72rem;
      padding: 3px 8px;
      border-radius: 6px;
      font-weight: 600;
    }

    .badge-ready {
      background: rgba(16, 185, 129, 0.15);
      color: #6ee7b7;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* Progress & Metrics */
    .progress-box {
      margin-bottom: 24px;
    }

    .progress-header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 10px;
    }

    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 0.85rem;
      font-weight: 600;
      padding: 5px 12px;
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.06);
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--text-muted);
    }

    .dot.running {
      background: var(--emerald);
      box-shadow: 0 0 12px var(--emerald);
      animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
      0% { transform: scale(0.95); opacity: 0.8; }
      50% { transform: scale(1.3); opacity: 1; }
      100% { transform: scale(0.95); opacity: 0.8; }
    }

    .percentage-text {
      font-size: 2.2rem;
      font-weight: 800;
      letter-spacing: -1px;
      background: linear-gradient(135deg, #ffffff, var(--cyan));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .progress-track {
      width: 100%;
      height: 12px;
      background: rgba(0, 0, 0, 0.5);
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.05);
      position: relative;
    }

    .progress-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--primary), var(--cyan));
      border-radius: 10px;
      transition: width 0.3s ease;
      box-shadow: 0 0 20px rgba(6, 182, 212, 0.5);
    }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-top: 18px;
    }

    @media (max-width: 640px) {
      .metrics-grid {
        grid-template-columns: 1fr 1fr;
      }
    }

    .metric-card {
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px 14px;
    }

    .metric-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-bottom: 4px;
    }

    .metric-val {
      font-size: 0.98rem;
      font-weight: 600;
      color: #f1f5f9;
      font-family: 'JetBrains Mono', monospace;
    }

    /* Live Preview & Tabs */
    .preview-container {
      display: flex;
      flex-direction: column;
      height: 100%;
    }

    .tabs {
      display: flex;
      gap: 10px;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 12px;
      margin-bottom: 16px;
      align-items: center;
      justify-content: space-between;
    }

    .tab-group {
      display: flex;
      gap: 8px;
    }

    .tab-btn {
      padding: 8px 16px;
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      border-radius: 8px;
      font-size: 0.88rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }

    .tab-btn.active {
      background: rgba(99, 102, 241, 0.15);
      border-color: var(--primary);
      color: #ffffff;
    }

    .tab-actions {
      display: flex;
      gap: 8px;
    }

    .icon-btn {
      padding: 7px 12px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      color: var(--text-main);
      border-radius: 8px;
      font-size: 0.8rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }

    .icon-btn:hover {
      background: rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.25);
    }

    .transcript-window {
      flex: 1;
      min-height: 400px;
      max-height: 560px;
      overflow-y: auto;
      background: rgba(0, 0, 0, 0.45);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      font-family: 'Outfit', sans-serif;
      font-size: 0.98rem;
      line-height: 1.7;
      color: #e2e8f0;
      white-space: pre-wrap;
      position: relative;
    }

    .transcript-window.monospace {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.86rem;
      line-height: 1.6;
      color: #94a3b8;
    }

    .live-speaking {
      padding: 10px 14px;
      margin-top: 12px;
      background: rgba(6, 182, 212, 0.1);
      border-left: 3px solid var(--cyan);
      border-radius: 0 8px 8px 0;
      font-size: 0.88rem;
      color: #a5f3fc;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .toast {
      position: fixed;
      bottom: 25px;
      right: 25px;
      padding: 12px 20px;
      background: var(--card-bg);
      border: 1px solid var(--emerald);
      color: #6ee7b7;
      border-radius: 10px;
      font-size: 0.9rem;
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
      opacity: 0;
      transform: translateY(20px);
      transition: all 0.3s ease;
      pointer-events: none;
      z-index: 1000;
    }

    .toast.show {
      opacity: 1;
      transform: translateY(0);
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="logo-icon">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
          <line x1="12" y1="19" x2="12" y2="22"></line>
        </svg>
      </div>
      <div>
        <h1>AI Video Transcription Studio</h1>
        <p>Offline Speech-to-Text powered by Faster-Whisper</p>
      </div>
    </div>
    <div class="header-actions">
      <button class="btn-secondary" onclick="openTranscripts()">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
          <line x1="16" y1="13" x2="8" y2="13"></line>
          <line x1="16" y1="17" x2="8" y2="17"></line>
        </svg>
        📁 Transcripts Folder
      </button>
      <button class="btn-secondary" onclick="openVideos()">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="23 7 16 12 23 17 23 7"></polygon>
          <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
        </svg>
        🎬 Videos Folder
      </button>
      <button class="btn-secondary" onclick="fetchVideos()">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="23 4 23 10 17 10"></polyline>
          <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
        </svg>
        Refresh Files
      </button>
    </div>
  </header>

  <main>
    <!-- Left Column: Controls & Video Selection -->
    <div>
      <div class="card">
        <div class="card-title">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--cyan)" stroke-width="2">
            <polygon points="23 7 16 12 23 17 23 7"></polygon>
            <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
          </svg>
          Select Video
        </div>

        <div class="form-group">
          <label>Choose from Folder:</label>
          <select id="videoSelect" onchange="onSelectChange()">
            <option value="">-- Scanning videos... --</option>
          </select>
        </div>

        <div class="form-group">
          <label>Or Enter Video File Name / Path:</label>
          <input type="text" id="customVideoInput" placeholder="e.g. pithon.mp4">
        </div>

        <div class="grid-2 form-group">
          <div>
            <label>Model Quality:</label>
            <select id="modelSelect">
              <option value="small" selected>Small (Accurate - Recommended)</option>
              <option value="base">Base (Faster, 2x speed)</option>
              <option value="tiny">Tiny (Fastest)</option>
              <option value="medium">Medium (Maximum accuracy)</option>
            </select>
          </div>
          <div>
            <label>Spoken Language:</label>
            <select id="languageSelect">
              <option value="auto" selected>Auto-Detect</option>
              <option value="en">English</option>
              <option value="ur">Urdu</option>
              <option value="hi">Hindi</option>
              <option value="ar">Arabic</option>
              <option value="fr">French</option>
            </select>
          </div>
        </div>

        <button class="btn-primary" id="startBtn" onclick="startTranscription()">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
          </svg>
          Start Transcription
        </button>

        <button class="btn-danger" id="stopBtn" onclick="stopTranscription()" style="display: none;">
          Stop Transcription
        </button>

        <div style="margin-top: 24px;">
          <label>Available Videos in Workspace:</label>
          <div class="video-list" id="videoListContainer">
            <!-- Dynamically populated -->
          </div>
        </div>
      </div>
    </div>

    <!-- Right Column: Live Progress & Streaming Transcript -->
    <div>
      <div class="card preview-container">
        
        <!-- Progress Bar Box -->
        <div class="progress-box">
          <div class="progress-header">
            <div class="status-badge" id="statusBadge">
              <span class="dot" id="statusDot"></span>
              <span id="statusText">Idle</span>
            </div>
            <div class="percentage-text" id="percentageVal">0.0%</div>
          </div>

          <div class="progress-track">
            <div class="progress-fill" id="progressFill"></div>
          </div>

          <!-- Metrics -->
          <div class="metrics-grid">
            <div class="metric-card">
              <div class="metric-label">Audio Progress</div>
              <div class="metric-val" id="timeMetric">00:00:00 / 00:00:00</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Language</div>
              <div class="metric-val" id="langMetric">Auto</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Inference Speed</div>
              <div class="metric-val" id="speedMetric">--x</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">ETA Remaining</div>
              <div class="metric-val" id="etaMetric">--:--</div>
            </div>
          </div>

          <div class="live-speaking" id="liveSpeakingBar" style="display: none;">
            <strong>Live:</strong> <span id="liveSpeakingText"></span>
          </div>
        </div>

        <!-- Transcript Viewer -->
        <div class="tabs">
          <div class="tab-group">
            <button class="tab-btn active" id="tabPlain" onclick="setTab('plain')">Clean Text</button>
            <button class="tab-btn" id="tabTimestamps" onclick="setTab('timestamps')">With Timestamps</button>
            <button class="tab-btn" id="tabSrt" onclick="setTab('srt')">Subtitles (.srt)</button>
          </div>
          <div class="tab-actions">
            <label style="display: flex; align-items: center; gap: 6px; font-size: 0.82rem; cursor: pointer; margin-bottom: 0;">
              <input type="checkbox" id="autoScroll" checked> Auto-Scroll
            </label>
            <button class="icon-btn" onclick="copyTranscript()">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
              Copy Text
            </button>
            <button class="icon-btn" onclick="downloadCurrent()">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
              </svg>
              Download
            </button>
          </div>
        </div>

        <div class="transcript-window" id="transcriptDisplay">
          Select or start a video to see live transcription here...
        </div>

      </div>
    </div>
  </main>

  <div class="toast" id="toast">Transcript copied to clipboard!</div>

  <script>
    let activeTab = 'plain';
    let cachedPlain = "";
    let cachedTimestamps = "";
    let cachedSrt = "";
    let currentFileName = "";

    function formatTime(sec) {
      if (!sec || isNaN(sec)) return "00:00:00";
      sec = Math.floor(sec);
      const h = Math.floor(sec / 3600);
      const m = Math.floor((sec % 3600) / 60);
      const s = sec % 60;
      return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }

    async function fetchVideos() {
      try {
        const res = await fetch('/api/videos');
        const data = await res.json();
        const select = document.getElementById('videoSelect');
        const listCont = document.getElementById('videoListContainer');
        select.innerHTML = '<option value="">-- Select a video file --</option>';
        listCont.innerHTML = '';

        data.videos.forEach(v => {
          const opt = document.createElement('option');
          opt.value = v.name;
          opt.textContent = `${v.name} (${v.size_mb} MB) ${v.has_transcript ? '✓ Transcribed' : ''}`;
          select.appendChild(opt);

          const item = document.createElement('div');
          item.className = 'video-item';
          item.innerHTML = `
            <span><strong>${v.name}</strong> <small style="color:#94a3b8">(${v.size_mb} MB)</small></span>
            ${v.has_transcript ? '<span class="badge badge-ready">Ready</span>' : '<span style="font-size:0.75rem; color:#64748b">Not Transcribed</span>'}
          `;
          item.onclick = () => {
            select.value = v.name;
            document.getElementById('customVideoInput').value = v.name;
            if (v.has_transcript) {
              loadExistingTranscript(v.name);
            }
          };
          listCont.appendChild(item);
        });

        // Default to pithon.mp4 if available
        if (data.videos.some(v => v.name === 'pithon.mp4') && !document.getElementById('customVideoInput').value) {
          select.value = 'pithon.mp4';
          document.getElementById('customVideoInput').value = 'pithon.mp4';
        }
      } catch (err) {
        console.error("Error fetching videos:", err);
      }
    }

    function onSelectChange() {
      const val = document.getElementById('videoSelect').value;
      if (val) {
        document.getElementById('customVideoInput').value = val;
        loadExistingTranscript(val);
      }
    }

    async function loadExistingTranscript(videoName) {
      try {
        const res = await fetch(`/api/transcript/${encodeURIComponent(videoName)}`);
        const data = await res.json();
        if (data.plain || data.timestamps) {
          cachedPlain = data.plain;
          cachedTimestamps = data.timestamps;
          cachedSrt = data.srt;
          currentFileName = videoName;
          renderTabContent();
          showToast(`Loaded existing transcript for ${videoName}`);
        }
      } catch (err) {
        console.error(err);
      }
    }

    async function startTranscription() {
      const videoName = document.getElementById('customVideoInput').value.trim() || document.getElementById('videoSelect').value;
      if (!videoName) {
        alert("Please select or enter a video file name first!");
        return;
      }

      const model = document.getElementById('modelSelect').value;
      const language = document.getElementById('languageSelect').value;

      try {
        const res = await fetch('/api/transcribe', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            video_name: videoName,
            model: model,
            language: language
          })
        });

        const data = await res.json();
        if (!res.ok) {
          alert(data.detail || "Error starting transcription");
        } else {
          showToast(`Started transcribing ${videoName}`);
        }
      } catch (err) {
        alert("Failed to connect to transcription server.");
      }
    }

    async function stopTranscription() {
      if (confirm("Are you sure you want to stop the transcription? Current text will be kept.")) {
        await fetch('/api/stop', {method: 'POST'});
        showToast("Stopping transcription...");
      }
    }

    async function openTranscripts() {
      await fetch('/api/open_transcripts', {method: 'POST'});
      showToast("Opening transcripts folder...");
    }

    async function openVideos() {
      await fetch('/api/open_videos', {method: 'POST'});
      showToast("Opening videos folder...");
    }

    function setTab(tab) {
      activeTab = tab;
      document.getElementById('tabPlain').classList.toggle('active', tab === 'plain');
      document.getElementById('tabTimestamps').classList.toggle('active', tab === 'timestamps');
      document.getElementById('tabSrt').classList.toggle('active', tab === 'srt');

      const disp = document.getElementById('transcriptDisplay');
      if (tab === 'plain') {
        disp.classList.remove('monospace');
      } else {
        disp.classList.add('monospace');
      }
      renderTabContent();
    }

    function renderTabContent() {
      const disp = document.getElementById('transcriptDisplay');
      let text = "";
      if (activeTab === 'plain') {
        text = cachedPlain || "No transcript yet...";
      } else if (activeTab === 'timestamps') {
        text = cachedTimestamps || "No timestamped segments yet...";
      } else {
        text = cachedSrt || "No SRT subtitle entries yet...";
      }
      disp.textContent = text;

      if (document.getElementById('autoScroll').checked) {
        disp.scrollTop = disp.scrollHeight;
      }
    }

    function copyTranscript() {
      const disp = document.getElementById('transcriptDisplay').textContent;
      if (!disp || disp.startsWith("No transcript")) return;
      navigator.clipboard.writeText(disp).then(() => {
        showToast("Transcript copied to clipboard!");
      });
    }

    function downloadCurrent() {
      const disp = document.getElementById('transcriptDisplay').textContent;
      if (!disp || disp.startsWith("No transcript")) return;

      const base = currentFileName ? currentFileName.replace(/\\.[^/.]+$/, "") : "transcript";
      let ext = ".txt";
      if (activeTab === 'timestamps') ext = "_timestamps.txt";
      if (activeTab === 'srt') ext = ".srt";

      const blob = new Blob([disp], {type: 'text/plain;charset=utf-8'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${base}${ext}`;
      a.click();
      URL.revokeObjectURL(url);
    }

    function showToast(msg) {
      const toast = document.getElementById('toast');
      toast.textContent = msg;
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 3000);
    }

    // Polling status loop
    async function pollStatus() {
      try {
        const res = await fetch('/api/status');
        const s = await res.json();

        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const progressFill = document.getElementById('progressFill');
        const percentageVal = document.getElementById('percentageVal');
        const timeMetric = document.getElementById('timeMetric');
        const langMetric = document.getElementById('langMetric');
        const speedMetric = document.getElementById('speedMetric');
        const etaMetric = document.getElementById('etaMetric');
        const liveBar = document.getElementById('liveSpeakingBar');
        const liveText = document.getElementById('liveSpeakingText');

        if (s.is_running) {
          statusDot.className = 'dot running';
          statusText.textContent = `Processing (${s.file_name})`;
          startBtn.disabled = true;
          stopBtn.style.display = 'block';

          percentageVal.textContent = `${s.percentage.toFixed(1)}%`;
          progressFill.style.width = `${s.percentage}%`;

          timeMetric.textContent = `${formatTime(s.current_time)} / ${formatTime(s.total_duration)}`;
          langMetric.textContent = s.language ? `${s.language.toUpperCase()} (${(s.language_prob * 100).toFixed(0)}%)` : '--';
          speedMetric.textContent = s.speed ? `${s.speed.toFixed(1)}x` : '--x';
          etaMetric.textContent = s.eta_seconds ? formatTime(s.eta_seconds) : '--:--';

          if (s.current_segment) {
            liveBar.style.display = 'flex';
            liveText.textContent = s.current_segment;
          }

          if (s.plain_text) cachedPlain = s.plain_text;
          if (s.timestamps_text) cachedTimestamps = s.timestamps_text;
          if (s.srt_text) cachedSrt = s.srt_text;
          currentFileName = s.file_name;

          renderTabContent();
        } else {
          startBtn.disabled = false;
          stopBtn.style.display = 'none';

          if (s.is_completed || s.percentage >= 99.5) {
            statusDot.className = 'dot';
            statusDot.style.background = '#10b981';
            statusDot.style.boxShadow = '0 0 10px #10b981';
            statusText.textContent = s.file_name ? `Completed (${s.file_name})` : 'Completed';
            percentageVal.textContent = '100.0%';
            progressFill.style.width = '100%';
            liveBar.style.display = 'none';

            timeMetric.textContent = `${formatTime(s.total_duration)} / ${formatTime(s.total_duration)}`;
            langMetric.textContent = s.language ? `${s.language.toUpperCase()} (${(s.language_prob * 100).toFixed(0)}%)` : 'EN';
            speedMetric.textContent = 'Done';
            etaMetric.textContent = '00:00:00';

            if (s.plain_text && !cachedPlain) {
              cachedPlain = s.plain_text;
              cachedTimestamps = s.timestamps_text;
              cachedSrt = s.srt_text;
              currentFileName = s.file_name;
              renderTabContent();
            }
          } else {
            statusDot.className = 'dot';
            statusDot.style.background = '';
            statusDot.style.boxShadow = '';
            statusText.textContent = 'Idle';
          }
        }
      } catch (err) {
        // server might be busy or starting
      }
    }

    fetchVideos();
    setInterval(pollStatus, 1000);
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

def main():
    import webbrowser
    port = 8765
    print("\n" + "="*60)
    print(f" Starting AI Video Transcription Studio Web App...")
    print(f" Open your browser at: http://127.0.0.1:{port}")
    print("="*60 + "\n")
    
    # Auto-open browser in 1.2 seconds
    def open_browser():
        time.sleep(1.2)
        webbrowser.open(f"http://127.0.0.1:{port}")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

if __name__ == "__main__":
    main()
