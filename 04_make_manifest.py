#!/usr/bin/env python3
"""
Step 4 – Build train / val / test manifests in NeMo JSONL format.

Reads all_segments.json produced by Step 3, applies quality filters, splits
the data by hearing (to prevent leakage), and writes three manifest files:

    data/manifests/train.jsonl
    data/manifests/val.jsonl
    data/manifests/test.jsonl

Each JSONL line:
    {"audio_filepath": "/abs/path/seg_00001.wav", "text": "…", "duration": 8.3}

This format is compatible with NeMo, HuggingFace ASR datasets, ESPnet,
k2/icefall, and most other ASR frameworks.

Why split by hearing?
    If segments from the same court session appear in both train and test the
    model can memorise specific speaker styles / case vocabulary, inflating
    test scores.  Assigning full hearings to one split prevents this.

Usage:
    python 04_make_manifest.py \\
        --segments-json ./data/segments/all_segments.json \\
        --out-dir ./data/manifests
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    """
    Final text clean-up before writing to manifests:
      • Remove bracket-enclosed noise markers  [UNCLEAR], (laughs), etc.
      • Strip control characters.
      • Collapse internal whitespace.
    """
    text = re.sub(r"\[.*?\]", "", text)       # [UNCLEAR] / [INAUDIBLE]
    text = re.sub(r"\(.*?\)", "", text)        # (laughter) / (pause)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_segments(
    segments: list[dict],
    min_duration: float,
    max_duration: float,
    min_words: int,
    min_match_score: float,
) -> tuple[list[dict], dict[str, int]]:
    """
    Apply quality filters.  Returns (kept_records, drop_counts).

    Each kept record contains:
        audio_filepath, text, duration, hearing
    The hearing field is preserved for the split step.
    """
    kept: list[dict] = []
    dropped: dict[str, int] = {
        "missing_file": 0,
        "duration": 0,
        "match_score": 0,
        "empty_text": 0,
        "min_words": 0,
    }

    for seg in segments:
        # Audio file must exist on disk
        audio_path = Path(seg["audio_path"])
        if not audio_path.exists():
            dropped["missing_file"] += 1
            continue

        dur = float(seg.get("duration", 0.0))
        if not (min_duration <= dur <= max_duration):
            dropped["duration"] += 1
            continue

        score = float(seg.get("match_score", 100.0))
        if score < min_match_score:
            dropped["match_score"] += 1
            continue

        text = normalise_text(str(seg.get("text", "")))
        if not text:
            dropped["empty_text"] += 1
            continue
        if len(text.split()) < min_words:
            dropped["min_words"] += 1
            continue

        kept.append({
            "audio_filepath": str(audio_path.resolve()),
            "text":           text,
            "duration":       round(dur, 3),
            "hearing":        seg.get("hearing", "unknown"),
        })

    return kept, dropped


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def split_by_hearing(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Assign entire hearings to one of {train, val, test}.

    Strategy:
      1. Collect unique hearing slugs.
      2. Shuffle them (deterministically via seed).
      3. Assign the first n_val slugs → val, next n_test → test, rest → train.

    This completely prevents cross-split contamination.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 1e-6, "train_ratio + val_ratio must be < 1.0"

    hearings = sorted({r["hearing"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(hearings)

    n = len(hearings)
    n_val  = max(1, round(n * val_ratio))
    n_test = max(1, round(n * test_ratio))
    # Guard: never let val+test consume all hearings
    while n_val + n_test >= n and n >= 3:
        n_test = max(1, n_test - 1)

    val_set   = set(hearings[:n_val])
    test_set  = set(hearings[n_val: n_val + n_test])
    train_set = set(hearings[n_val + n_test:])

    train = [r for r in records if r["hearing"] in train_set]
    val   = [r for r in records if r["hearing"] in val_set]
    test  = [r for r in records if r["hearing"] in test_set]

    print(f"\n  Hearing split:")
    print(f"    train : {len(train_set)} hearings → {len(train):,} segments")
    print(f"    val   : {len(val_set)}   hearings → {len(val):,} segments")
    print(f"    test  : {len(test_set)}  hearings → {len(test):,} segments")

    return train, val, test


# ---------------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------------

def write_manifest(records: list[dict], path: Path) -> None:
    """
    Write records to a JSONL file.
    Only the standard ASR manifest fields (audio_filepath, text, duration)
    are written — the 'hearing' field is used only for splitting, not stored.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            entry = {
                "audio_filepath": rec["audio_filepath"],
                "text":           rec["text"],
                "duration":       rec["duration"],
            }
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total_dur = sum(r["duration"] for r in records)
    print(f"  {path.name:<20s}: {len(records):6,} segments  ({total_dur/3600:.2f} h)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build NeMo-format train/val/test manifests from aligned segments."
    )
    ap.add_argument(
        "--segments-json", default="./data/segments/all_segments.json",
        help="Path to all_segments.json produced by 03_align_segment.py",
    )
    ap.add_argument("--out-dir", default="./data/manifests")

    # Quality filters
    ap.add_argument("--min-duration",    type=float, default=3.0,
                    help="Minimum segment duration in seconds.")
    ap.add_argument("--max-duration",    type=float, default=25.0,
                    help="Maximum segment duration in seconds.")
    ap.add_argument("--min-words",       type=int,   default=4,
                    help="Minimum word count in transcript.")
    ap.add_argument("--min-match-score", type=float, default=60.0,
                    help="Minimum fuzzy match score (0–100).")

    # Split ratios
    ap.add_argument("--train-ratio", type=float, default=0.90)
    ap.add_argument("--val-ratio",   type=float, default=0.05)
    ap.add_argument("--seed",        type=int,   default=42)

    args = ap.parse_args()

    seg_path = Path(args.segments_json)
    if not seg_path.exists():
        raise FileNotFoundError(
            f"Segments index not found: {seg_path}\n"
            "Run 03_align_segment.py first."
        )

    with seg_path.open(encoding="utf-8") as fh:
        all_segments = json.load(fh)

    print(f"Loaded {len(all_segments):,} raw segments.")
    print("\nApplying quality filters …")

    kept, dropped = filter_segments(
        all_segments,
        min_duration    = args.min_duration,
        max_duration    = args.max_duration,
        min_words       = args.min_words,
        min_match_score = args.min_match_score,
    )

    print(f"  Kept   : {len(kept):,}")
    for reason, count in dropped.items():
        if count:
            print(f"  Dropped ({reason}) : {count:,}")

    if not kept:
        raise RuntimeError(
            "No segments passed filters.  "
            "Try lowering --min-match-score or --min-duration."
        )

    train, val, test = split_by_hearing(
        kept,
        train_ratio = args.train_ratio,
        val_ratio   = args.val_ratio,
        seed        = args.seed,
    )

    out_dir = Path(args.out_dir)
    print("\nWriting manifests:")
    write_manifest(train, out_dir / "train.jsonl")
    write_manifest(val,   out_dir / "val.jsonl")
    write_manifest(test,  out_dir / "test.jsonl")

    total_h = sum(r["duration"] for r in kept) / 3600
    print(f"\nTotal usable data : {total_h:.2f} h  in {len(kept):,} segments")
    print(f"Manifests written : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
