#!/usr/bin/env python3
"""
Step 2 – Parse TERES-format Supreme Court transcript PDFs into clean text.

TERES transcript format
-----------------------
Each page looks like:

    Transcribed by TERES                          (header – drop)
      12                                          (page number – drop)

    CHIEF JUSTICE D. Y. CHANDRACHUD: We have made it very    1
    clear we are dealing with the validity of section six a   2

    SANJAY HEGDE: My Lords, just a prelude before I begin.   3

Rules:
  • Lines starting with "SPEAKER NAME: text" begin a new speaker turn.
    Speaker labels are ALL-CAPS words that may contain dots, apostrophes,
    hyphens, and spaces.
  • Lines that do NOT match a speaker label continue the previous turn.
  • Trailing right-margin numbers (line counters) are stripped.
  • TERES page headers, timestamp lines, and standalone numbers are discarded.
  • [UNCLEAR] / [INAUDIBLE] markers are removed from the final text.

Outputs per hearing directory:
  utterances.json  – list of {"speaker": ..., "text": ...}
  utterances.txt   – one clean utterance per line (used by alignment step)

Usage:
    python 02_parse_transcript.py --raw-dir ./data/raw
    python 02_parse_transcript.py --raw-dir ./data/raw --hearing 001_12.12.2023
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pypdf import PdfReader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Speaker label: all-caps word(s) optionally with dots/apostrophes/hyphens
# followed by a colon and the start of the utterance.
_SPEAKER_RE = re.compile(
    r"^([A-Z][A-Z0-9 .\-',]+?):\s+(.+)",
    re.DOTALL,
)

# Right-margin line numbers like "  27" or "  3"
_LINE_NUM_SUFFIX_RE = re.compile(r"\s{2,}\d{1,3}\s*$")

# Standalone page or line numbers on their own line
_STANDALONE_NUM_RE = re.compile(r"^\d+\s*$")

# TERES page header
_TERES_HEADER_RE = re.compile(r"transcribed\s+by\s+teres", re.IGNORECASE)

# Timestamp lines: "11:05 AM IST", "9:30 AM IST", etc.
_TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}\s+[AP]M\s+IST$", re.IGNORECASE)

# Noise markers to scrub from final text
_NOISE_RE = re.compile(r"\[(?:UNCLEAR|INAUDIBLE|CROSSTALK)[^\]]*\]", re.IGNORECASE)


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_raw_lines(pdf_path: Path) -> list[str]:
    """
    Read every page of a TERES PDF and return a flat list of non-empty,
    non-artefact text lines.
    """
    reader = PdfReader(str(pdf_path))
    lines: list[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if _TERES_HEADER_RE.search(line):
                continue
            if _TIMESTAMP_RE.match(line):
                continue
            if _STANDALONE_NUM_RE.match(line):
                continue
            lines.append(line)

    return lines


# ---------------------------------------------------------------------------
# Utterance parsing
# ---------------------------------------------------------------------------

def parse_utterances(raw_lines: list[str]) -> list[dict]:
    """
    Turn a flat list of PDF lines into a list of speaker-utterance dicts:
        {"speaker": "SANJAY HEGDE", "text": "My Lords, just a prelude …"}

    Multi-line utterances are joined; continuation lines (those not matching
    a speaker label) are appended to the current speaker's text.
    """
    utterances: list[dict] = []
    current_speaker: str | None = None
    current_parts: list[str] = []

    def flush() -> None:
        if current_speaker and current_parts:
            raw = " ".join(current_parts)
            cleaned = _clean_utterance(raw)
            if cleaned:
                utterances.append({"speaker": current_speaker, "text": cleaned})

    for line in raw_lines:
        # Strip right-margin line numbers
        line = _LINE_NUM_SUFFIX_RE.sub("", line).strip()
        if not line:
            continue

        m = _SPEAKER_RE.match(line)
        if m:
            flush()
            current_speaker = m.group(1).strip()
            current_parts   = [m.group(2).strip()]
        elif current_speaker is not None:
            current_parts.append(line)

    flush()
    return utterances


def _clean_utterance(text: str) -> str:
    """Remove noise markers, stray numbers, and extra whitespace."""
    text = _NOISE_RE.sub("", text)
    # Strip residual right-margin numbers that survived earlier
    text = re.sub(r"\s+\d{1,3}\s*$", "", text)
    # Collapse internal whitespace
    text = re.sub(r"\s+", " ", text)
    # Drop leading/trailing punctuation artefacts (but keep sentence-final periods)
    text = text.strip(" .,;:-")
    return text.strip()


# ---------------------------------------------------------------------------
# Per-hearing processing
# ---------------------------------------------------------------------------

def process_hearing(hearing_dir: Path) -> int:
    """
    Parse the transcript PDF in hearing_dir.
    Writes utterances.json and utterances.txt.
    Returns the number of utterances extracted.
    """
    pdf_path = hearing_dir / "transcript.pdf"
    if not pdf_path.exists():
        print(f"  [SKIP] No PDF in {hearing_dir.name}")
        return 0

    raw_lines  = extract_raw_lines(pdf_path)
    utterances = parse_utterances(raw_lines)

    if not utterances:
        print(f"  [WARN] No utterances extracted from {hearing_dir.name}")
        return 0

    # JSON: preserves speaker labels for downstream analysis
    json_path = hearing_dir / "utterances.json"
    json_path.write_text(
        json.dumps(utterances, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Plain text: one utterance per line, used by the alignment step
    txt_path = hearing_dir / "utterances.txt"
    txt_path.write_text(
        "\n".join(u["text"] for u in utterances), encoding="utf-8"
    )

    return len(utterances)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse TERES Supreme Court PDFs into clean utterance files."
    )
    ap.add_argument("--raw-dir",  default="./data/raw",
                    help="Directory containing one sub-folder per hearing.")
    ap.add_argument("--hearing",  default="",
                    help="Process only this hearing slug (e.g. 001_12.12.2023).")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    if args.hearing:
        hearing_dirs = [raw_dir / args.hearing]
    else:
        hearing_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())

    total = 0
    for d in tqdm(hearing_dirs, desc="Parsing PDFs"):
        n = process_hearing(d)
        total += n
        if n:
            print(f"  {d.name}: {n} utterances")

    print(f"\nDone.  Total utterances: {total}")
    print("Output: utterances.txt + utterances.json in each hearing directory.")


if __name__ == "__main__":
    main()
