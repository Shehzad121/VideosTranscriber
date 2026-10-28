# VideosTranscriber

Automated, high-performance offline speech-to-text transcription studio for video and audio files powered by **faster-whisper**.

---

## 🌟 Key Features

- **Direct In-Memory Processing**: Transcribes `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.mp3`, `.wav`, and `.m4a` directly without creating temporary audio files on disk.
- **Multiple Output Formats**: Automatically produces 3 formats per video:
  - `_transcript.txt`: Clean, human-readable paragraphs.
  - `_timestamps.txt`: Precise timecoded segments (`[HH:MM:SS -> HH:MM:SS] Text`).
  - `.srt`: Standard subtitle format for media players and video editors.
- **Interactive Web Studio**: Sleek, modern browser interface with a real-time glowing progress bar, ETA, inference speed metrics, and live streaming transcript.
- **Dedicated Folder Structure**: Organizes each video's transcripts into its own subfolder inside `transcripts/<video_name>/`.
- **Zero API Fees / 100% Offline**: Runs locally using quantized CTranslate2 models (`int8` on CPU).
- **Voice Activity Detection (VAD)**: Automatically skips background silence to minimize processing time.

---

## 📁 Repository Structure

```text
VideosTranscriber/
│
├── 📂 videos/               # Place all input video/audio files here (git-ignored)
│   └── example_video.mp4
│
├── 📂 transcripts/          # Transcripts generated here, organized per video (git-ignored)
│   └── example_video/
│       ├── example_video_transcript.txt
│       ├── example_video_timestamps.txt
│       └── example_video.srt
│
├── 🚀 run_app.bat           # 1-Click double-click launcher for Windows
├── 🐍 app.py                # FastAPI backend & web studio UI server
├── 🛠️ transcribe.py         # Standalone CLI / batch processing script
├── 📋 requirements.txt      # Python dependencies
├── 📄 .gitignore            # Ignores raw media files, transcripts, and caches
├── ⚖️ LICENSE               # MIT License
└── 📖 README.md             # Project documentation
```

---

## ⚙️ Installation & Requirements

### 1. Requirements
- Python 3.10+
- FFmpeg (installed and added to system `PATH`)

### 2. Clone the Repository
```bash
git clone https://github.com/Shehzad121/VideosTranscriber.git
cd VideosTranscriber
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Method 1: Interactive Web Studio (Recommended)

1. **On Windows**: Double-click **`run_app.bat`**.
2. **Or via terminal**:
   ```bash
   python app.py
   ```
3. Your browser will automatically open at:
   ```
   http://127.0.0.1:8765
   ```
4. Select your video from the dropdown (or enter a file name), pick your model and language preferences, and click **Start Transcription**.

---

### Method 2: Command Line (CLI)

You can also run batch or single-video transcriptions from the terminal:

```bash
# 1. Transcribe a specific video:
python transcribe.py videos/my_video.mp4

# 2. Transcribe using a specific model (e.g., base for 2x faster execution):
python transcribe.py videos/my_video.mp4 --model base

# 3. Transcribe with explicit language code:
python transcribe.py videos/my_video.mp4 --language en

# 4. Transcribe all videos in the videos/ folder automatically:
python transcribe.py
```

---

## 📊 Available Whisper Models

| Model | Size | Speed (CPU) | Recommended For |
|---|---|---|---|
| `tiny` | ~75 MB | ~15x | Fast draft / testing |
| `base` | ~145 MB | ~10x | Fast processing with good accuracy |
| `small` (Default) | ~480 MB | ~5x | Optimal balance of high accuracy & speed |
| `medium` | ~1.5 GB | ~2x | Complex multi-speaker / heavy accents |
| `large-v3` | ~3.1 GB | ~1x | Maximum precision |

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
