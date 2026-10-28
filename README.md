# AI Video Transcription Studio

Clean, automated offline speech-to-text transcription for videos and audio powered by `faster-whisper`.

---

## 📁 Organized Project Structure

```text
AI4HealthVideos/
│
├── 📂 videos/               # 🎬 APNI TAMAAM VIDEOS YAHAAN RAKHEIN
│   ├── pithon.mp4
│   └── jgq-btet-sbi-2026-10-20.mp4
│
├── 📂 transcripts/          # 📄 TAMAAM TRANSCRIPTS YAHAAN AUTO-SAVE HONGI
│   ├── pithon_transcript.txt
│   ├── pithon_timestamps.txt
│   └── pithon.srt
│
├── 🚀 run_app.bat           # 1-Click Double-Click Web Launcher
├── 🐍 app.py                # Web Studio Backend & UI Server
├── 🛠️ transcribe.py         # CLI / Terminal Batch Script
└── 📖 README.md             # Project Guide
```

---

## 🚀 How to Use

### Method 1: Interactive Web Studio (Easiest)
1. Double click **`run_app.bat`** (or run `python app.py`).
2. Browser automatically opens at **`http://127.0.0.1:8765`**.
3. Select any video from the list (it automatically detects all videos inside `videos/`).
4. Click **Start Transcription** to see live progress, audio metrics, and streaming text.
5. Use **📁 Transcripts Folder** or **🎬 Videos Folder** buttons in the header to quickly access your files.

---

### Method 2: Command Line (CLI)
You can also transcribe directly from terminal / PowerShell:

```powershell
# 1. Transcribe any video inside videos/ folder:
python transcribe.py videos/pithon.mp4

# 2. Or simply give video name (it checks videos/ automatically):
python transcribe.py "jgq-btet-sbi-2026-10-20.mp4"

# 3. Transcribe ALL videos in the videos folder at once:
python transcribe.py
```

All output files (`.txt`, `_timestamps.txt`, `.srt`) are automatically placed into the `transcripts/` folder.
