#!/usr/bin/env python3
"""
Step 3 – Align long audio recordings with reference transcripts and segment
         them into short, training-ready WAV chunks.

Strategy
--------
Each hearing is a long audio file (18 – 345 min) paired with a reference
transcript extracted in Step 2.  We need short (3–25 s) audio segments with
accurate text labels.

We use WhisperX (https://github.com/m-bain/whisperX) which provides:
  1. Fast transcription via faster-whisper (batched inference) → rough timestamps
  2. Phoneme-level forced alignment via wav2vec2 → accurate word timestamps

Per hearing:
  a. Load 16 kHz mono WAV (produced by Step 1).
  b. Run WhisperX → list of word-level timestamps.
  c. Group words into chunks (≤ MAX_CHUNK_SEC) splitting on silence gaps or
     max-duration overflow.
  d. Fuzzy-match each chunk against the reference transcript to filter noise.
  e. Save each accepted chunk as an individual WAV segment.
  f. Write a segments.json listing path / text / duration / match_score.

A global all_segments.json is written to --out-dir.

Usage:
    # All hearings, GPU:
    python 03_align_segment.py --raw-dir ./data/raw --out-dir ./data/segments

    # Single hearing, for testing:
    python 03_align_segment.py --raw-dir ./data/raw --out-dir ./data/segments \
        --hearing 001_12.12.2023

    # CPU (very slow, only for testing):
    python 03_align_segment.py --raw-dir ./data/raw --out-dir ./data/segments \
        --device cpu --compute-type float32

Requirements:
    pip install whisperx soundfile rapidfuzz
    brew install ffmpeg   (macOS)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from rapidfuzz import fuzz
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Lazy WhisperX import (lets --help work without GPU)
# ---------------------------------------------------------------------------

_whisperx = None


def _wx():
    global _whisperx
    if _whisperx is None:
        try:
            import whisperx as wx
            _whisperx = wx
        except ImportError:
            sys.exit(
                "whisperx is not installed.\n"
                "Install it with:\n"
                "  pip install git+https://github.com/m-bain/whisperX.git\n"
                "  pip install whisperx"
            )
    return _whisperx


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE       = 16_000
MIN_CHUNK_SEC     = 3.0    # drop segments shorter than this
MAX_CHUNK_SEC     = 25.0   # split before this duration
MIN_WORDS         = 4      # drop segments with fewer words than this
SILENCE_GAP_SEC   = 1.5    # inter-word gap that triggers a chunk boundary


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_wav(wav_path: Path) -> np.ndarray:
    """Load a 16 kHz mono WAV as float32 numpy array."""
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Expected 16 kHz WAV, got {sr} Hz.  "
            f"Re-run 01_download.py to re-convert {wav_path}."
        )
    return audio


def write_wav_segment(audio: np.ndarray, start_s: float, end_s: float, out_path: Path) -> None:
    """Slice and write a segment of audio to disk as 16-bit PCM WAV."""
    s = max(0, int(start_s * SAMPLE_RATE))
    e = min(len(audio), int(end_s   * SAMPLE_RATE))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio[s:e], SAMPLE_RATE, subtype="PCM_16")


# ---------------------------------------------------------------------------
# WhisperX transcription + word alignment
# ---------------------------------------------------------------------------

def transcribe_and_align(
    audio: np.ndarray,
    model,
    align_model,
    align_metadata,
    device: str,
    batch_size: int = 8,
) -> list[dict]:
    """
    Run WhisperX transcription followed by forced phoneme-level alignment.

    Returns a list of word dicts:
        {"word": str, "start": float, "end": float}
    Words without valid timestamps are discarded.
    """
    wx = _wx()

    result         = model.transcribe(audio, batch_size=batch_size)
    result_aligned = wx.align(
        result["segments"], align_model, align_metadata, audio, device
    )

    words = []
    for seg in result_aligned.get("word_segments", []):
        if "start" in seg and "end" in seg and "word" in seg:
            w = seg["word"].strip()
            if w:
                words.append({
                    "word":  w,
                    "start": float(seg["start"]),
                    "end":   float(seg["end"]),
                })
    return words


# ---------------------------------------------------------------------------
# Word → chunk grouping
# ---------------------------------------------------------------------------

def group_into_chunks(
    words: list[dict],
    max_sec: float = MAX_CHUNK_SEC,
    silence_gap: float = SILENCE_GAP_SEC,
) -> list[dict]:
    """
    Group consecutive word dicts into utterance chunks.

    A new chunk begins when:
      • the gap between the previous word's end and the current word's start
        exceeds `silence_gap` (natural sentence boundary), OR
      • the chunk duration would exceed `max_sec`.

    Returns a list of {"start": float, "end": float, "text": str}.
    """
    if not words:
        return []

    chunks: list[dict] = []
    chunk_words = [words[0]]
    chunk_start = words[0]["start"]

    for prev, curr in zip(words, words[1:]):
        gap       = curr["start"] - prev["end"]
        chunk_dur = curr["end"]   - chunk_start

        if gap > silence_gap or chunk_dur > max_sec:
            chunks.append({
                "start": chunk_start,
                "end":   prev["end"],
                "text":  " ".join(w["word"] for w in chunk_words).strip(),
            })
            chunk_words = [curr]
            chunk_start = curr["start"]
        else:
            chunk_words.append(curr)

    if chunk_words:
        chunks.append({
            "start": chunk_start,
            "end":   chunk_words[-1]["end"],
            "text":  " ".join(w["word"] for w in chunk_words).strip(),
        })

    return chunks


# ---------------------------------------------------------------------------
# Fuzzy match quality score
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case and strip punctuation for fuzzy comparison."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def best_match_score(hypothesis: str, reference_lines: list[str]) -> float:
    """
    Return the highest token_set_ratio between `hypothesis` and any reference
    line.  Short-circuits at 95 since that is effectively a perfect match.
    """
    h_norm = _normalise(hypothesis)
    best = 0.0
    for ref in reference_lines:
        score = fuzz.token_set_ratio(h_norm, _normalise(ref))
        if score > best:
            best = score
        if best >= 95:
            break
    return best


# ---------------------------------------------------------------------------
# Per-hearing processing
# ---------------------------------------------------------------------------

def process_hearing(
    hearing_dir: Path,
    out_dir: Path,
    model,
    align_model,
    align_metadata,
    device: str,
    min_match: float,
    batch_size: int,
) -> list[dict]:
    """
    Align and segment one hearing.

    Returns a list of segment records:
        {
          "audio_path":   str (absolute),
          "text":         str,
          "duration":     float,
          "match_score":  float,
          "hearing":      str (slug),
          "segment_idx":  int,
        }
    """
    wav_path = hearing_dir / "audio.wav"
    txt_path = hearing_dir / "utterances.txt"

    if not wav_path.exists():
        print(f"  [SKIP] No WAV: {hearing_dir.name}")
        return []
    if not txt_path.exists():
        print(f"  [SKIP] No utterances.txt (run Step 2 first): {hearing_dir.name}")
        return []

    ref_lines = [
        line.strip()
        for line in txt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print(f"\n→ {hearing_dir.name}  ({len(ref_lines)} reference utterances)")

    audio    = load_wav(wav_path)
    duration = len(audio) / SAMPLE_RATE
    print(f"  Audio  : {duration/60:.1f} min")

    words = transcribe_and_align(audio, model, align_model, align_metadata, device, batch_size)
    print(f"  Words  : {len(words)}")

    chunks = group_into_chunks(words, max_sec=MAX_CHUNK_SEC, silence_gap=SILENCE_GAP_SEC)
    print(f"  Chunks : {len(chunks)}")

    hearing_out = out_dir / hearing_dir.name
    hearing_out.mkdir(parents=True, exist_ok=True)

    segments: list[dict] = []
    n_kept = 0

    for idx, chunk in enumerate(chunks):
        dur  = chunk["end"] - chunk["start"]
        text = chunk["text"]

        if dur < MIN_CHUNK_SEC:
            continue
        if len(text.split()) < MIN_WORDS:
            continue

        score = best_match_score(text, ref_lines)
        if score < min_match:
            continue

        seg_path = hearing_out / f"seg_{idx:05d}.wav"
        write_wav_segment(audio, chunk["start"], chunk["end"], seg_path)

        segments.append({
            "audio_path":  str(seg_path.resolve()),
            "text":        text,
            "duration":    round(dur, 3),
            "match_score": round(score, 1),
            "hearing":     hearing_dir.name,
            "segment_idx": idx,
        })
        n_kept += 1

    print(f"  Kept   : {n_kept} / {len(chunks)}")

    # Per-hearing index
    (hearing_out / "segments.json").write_text(
        json.dumps(segments, indent=2, ensure_ascii=False)
    )
    return segments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Align Supreme Court audio with transcripts using WhisperX."
    )
    ap.add_argument("--raw-dir",   default="./data/raw")
    ap.add_argument("--out-dir",   default="./data/segments")
    ap.add_argument("--hearing",   default="",
                    help="Process only this hearing slug.")
    ap.add_argument("--device",    default="cuda",
                    choices=["cuda", "cpu"])
    ap.add_argument("--whisper-model", default="large-v3",
                    choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"])
    ap.add_argument("--compute-type", default="float16",
                    choices=["float16", "int8", "float32"])
    ap.add_argument("--batch-size", type=int, default=8,
                    help="WhisperX transcription batch size.")
    ap.add_argument("--min-match-score", type=float, default=60.0,
                    help=(
                        "Minimum fuzzy match score (0–100) between WhisperX output "
                        "and reference transcript.  Lower = more data, more noise."
                    ))
    args = ap.parse_args()

    # CPU requires float32
    if args.device == "cpu" and args.compute_type == "float16":
        args.compute_type = "float32"
        print("Note: switched compute_type to float32 for CPU.")

    wx = _wx()

    print(f"Loading Whisper model '{args.whisper_model}' on {args.device} …")
    model = wx.load_model(
        args.whisper_model,
        device=args.device,
        compute_type=args.compute_type,
    )

    print("Loading forced-alignment model (English wav2vec2) …")
    align_model, align_metadata = wx.load_align_model(
        language_code="en",
        device=args.device,
    )

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.hearing:
        hearing_dirs = [raw_dir / args.hearing]
    else:
        hearing_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())

    all_segments: list[dict] = []
    for hdir in hearing_dirs:
        segs = process_hearing(
            hdir, out_dir,
            model, align_model, align_metadata,
            args.device, args.min_match_score, args.batch_size,
        )
        all_segments.extend(segs)

    global_index = out_dir / "all_segments.json"
    global_index.write_text(json.dumps(all_segments, indent=2, ensure_ascii=False))

    total_h = sum(s["duration"] for s in all_segments) / 3600
    print(f"\n{'='*56}")
    print(f"  Total segments : {len(all_segments)}")
    print(f"  Total duration : {total_h:.2f} h")
    print(f"  Index saved to : {global_index}")
    print(f"{'='*56}")


if __name__ == "__main__":
    main()
