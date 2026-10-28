"""
Video & Audio Transcription Tool
Powered by faster-whisper

Features:
- Transcribes MP4, MKV, AVI, MOV, WEBM, MP3, WAV, etc. directly without saving audio files.
- Generates 3 clean output formats:
    1. <name>_transcript.txt: Clean, paragraph-formatted transcript text.
    2. <name>_timestamps.txt: Timestamped text [HH:MM:SS -> HH:MM:SS].
    3. <name>.srt: Standard subtitle file.
- Real-time progress bar with percentage and estimated time remaining.
- Continuous streaming write (results saved incrementally as transcription proceeds).
- Multi-video and folder batch processing support.
"""

import os
import sys
import argparse
import datetime
from pathlib import Path
from tqdm import tqdm
from faster_whisper import WhisperModel

def format_timestamp(seconds: float, srt_format: bool = False) -> str:
    """Convert seconds to HH:MM:SS,mmm or HH:MM:SS."""
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)

    if srt_format:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def transcribe_file(
    file_path: str,
    model: WhisperModel,
    language: str = None,
    output_dir: str = None
) -> dict:
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        # Check if it exists in videos/ subfolder
        alt_path = file_path.parent / "videos" / file_path.name
        if alt_path.exists():
            file_path = alt_path
        else:
            print(f"[Error] File not found: {file_path}")
            return None

    if output_dir:
        out_path_dir = Path(output_dir).resolve()
    else:
        # Default to transcripts/ folder inside project
        project_root = file_path.parent if file_path.parent.name != "videos" else file_path.parent.parent
        out_path_dir = project_root / "transcripts"
    
    out_path_dir.mkdir(parents=True, exist_ok=True)

    base_name = file_path.stem
    video_out_dir = out_path_dir / base_name
    video_out_dir.mkdir(parents=True, exist_ok=True)

    txt_plain_path = video_out_dir / f"{base_name}_transcript.txt"
    txt_time_path = video_out_dir / f"{base_name}_timestamps.txt"
    srt_path = video_out_dir / f"{base_name}.srt"

    print(f"\n{'='*70}")
    print(f"Processing: {file_path.name}")
    print(f"Output Directory: {out_path_dir}")
    print(f"{'='*70}")

    # Transcribe directly from video without extracting audio to disk
    segments_gen, info = model.transcribe(
        str(file_path),
        language=language,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=5
    )

    total_duration = info.duration
    detected_lang = info.language
    lang_prob = info.language_probability

    print(f"-> Detected Language : {detected_lang.upper()} (Confidence: {lang_prob:.1%})")
    print(f"-> Total Audio Length: {format_timestamp(total_duration)} ({total_duration:.1f}s)")
    print(f"-> Starting Transcription...\n")

    # Open files for incremental writing
    with open(txt_plain_path, "w", encoding="utf-8") as f_plain, \
         open(txt_time_path, "w", encoding="utf-8") as f_time, \
         open(srt_path, "w", encoding="utf-8") as f_srt:

        # Header for timestamps file
        f_time.write(f"# Transcript: {file_path.name}\n")
        f_time.write(f"# Language: {detected_lang} | Duration: {format_timestamp(total_duration)}\n\n")

        pbar = tqdm(total=round(total_duration, 1), unit="s", desc="Transcribing", dynamic_ncols=True)
        current_time = 0.0
        prev_end_time = 0.0
        paragraph_word_count = 0
        srt_idx = 1
        all_segments = []

        try:
            for segment in segments_gen:
                text = segment.text.strip()
                if not text:
                    continue

                all_segments.append(segment)
                start_str = format_timestamp(segment.start)
                end_str = format_timestamp(segment.end)
                srt_start = format_timestamp(segment.start, srt_format=True)
                srt_end = format_timestamp(segment.end, srt_format=True)

                # Check pause between segments for natural paragraph breaks
                silence_gap = segment.start - prev_end_time
                words = text.split()
                paragraph_word_count += len(words)

                # 1. Plain text file (readable paragraphs)
                if (silence_gap > 2.0 or paragraph_word_count > 80) and prev_end_time > 0:
                    f_plain.write(f"\n\n{text}")
                    paragraph_word_count = len(words)
                else:
                    if prev_end_time > 0:
                        f_plain.write(f" {text}")
                    else:
                        f_plain.write(text)
                f_plain.flush()

                # 2. Timestamped format
                f_time.write(f"[{start_str} -> {end_str}] {text}\n")
                f_time.flush()

                # 3. SRT format
                f_srt.write(f"{srt_idx}\n{srt_start} --> {srt_end}\n{text}\n\n")
                f_srt.flush()
                srt_idx += 1

                # Update progress bar
                advance = max(0.0, segment.end - current_time)
                pbar.update(round(advance, 1))
                current_time = max(current_time, segment.end)
                prev_end_time = segment.end

            f_plain.write("\n")
            f_plain.flush()

            # Finish remaining progress if any
            if current_time < total_duration:
                pbar.update(round(total_duration - current_time, 1))
        finally:
            pbar.close()

    print(f"\n[Done] Successfully transcribed: {file_path.name}")
    print(f"  1. Plain Transcript : {txt_plain_path.name}")
    print(f"  2. With Timestamps  : {txt_time_path.name}")
    print(f"  3. Subtitles (.srt) : {srt_path.name}\n")

    return {
        "plain_text": str(txt_plain_path),
        "timestamps": str(txt_time_path),
        "srt": str(srt_path),
        "language": detected_lang,
        "duration": total_duration,
        "segment_count": len(all_segments)
    }

def main():
    parser = argparse.ArgumentParser(description="Extract clean transcripts from video/audio files using faster-whisper.")
    parser.add_argument("files", nargs="*", help="Path(s) to video or audio files. If not specified, defaults to pithon.mp4 if present.")
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"], help="Whisper model size (default: small)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Computation device (default: cpu)")
    parser.add_argument("--compute_type", default="int8", choices=["int8", "float32", "int8_float32"], help="Compute quantization (default: int8)")
    parser.add_argument("--language", default=None, help="Force audio language code (e.g. 'en', 'ur', 'hi', 'ar'). Default: auto-detect")
    parser.add_argument("--output_dir", default=None, help="Directory to save transcript files (default: same directory as video)")
    parser.add_argument("--batch_dir", default=None, help="Batch transcribe all video files in this directory")
    args = parser.parse_args()

    # Determine files to process
    valid_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".m4a"}
    files_to_process = []
    if args.batch_dir:
        batch_path = Path(args.batch_dir)
        files_to_process = [f for f in batch_path.iterdir() if f.suffix.lower() in valid_exts]
    elif args.files:
        for f in args.files:
            p = Path(f)
            if not p.exists() and (Path("videos") / f).exists():
                p = Path("videos") / f
            files_to_process.append(p)
    else:
        # Check videos/ folder first, then current directory
        videos_dir = Path("videos")
        if videos_dir.exists():
            files_to_process = sorted([f for f in videos_dir.iterdir() if f.suffix.lower() in valid_exts])
        if not files_to_process:
            files_to_process = sorted([f for f in Path(".").iterdir() if f.suffix.lower() in valid_exts])

    if not files_to_process:
        print("[Error] No video/audio files found to transcribe.")
        sys.exit(1)

    print(f"Loading Whisper model '{args.model}' on {args.device} (compute_type={args.compute_type})...")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print("Model loaded successfully!")

    print(f"Found {len(files_to_process)} file(s) to transcribe.")
    for f in files_to_process:
        transcribe_file(
            file_path=str(f),
            model=model,
            language=args.language,
            output_dir=args.output_dir
        )

    print("\nAll transcriptions finished successfully!")

if __name__ == "__main__":
    main()
