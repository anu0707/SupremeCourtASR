#!/usr/bin/env python3
"""
category_eval.py
=================
Per-category WER breakdown, meant to be imported into 06_evaluate.py.

Categories (matches your report template):
    - statute_citations           ("Section 6A", "Article 21", "Rule 5")
    - case_numbers                ("SLP 1234/2023", "W.P. 45/2021")
    - latin_legal_terms           (suo motu, locus standi, prima facie, ...)
    - proper_nouns_judges_advocates   (needs a name list — see below)
    - hindi_sanskrit_proper_nouns     (needs a name list — see below)
    - general_legal_argument      (fallback — everything else)

WHY NAME LISTS ARE REQUIRED
----------------------------
Regex can catch "Section 6A" or "suo motu" because those follow a fixed
lexical pattern. It CANNOT reliably catch "Chandrachud" or "Hegde" — there's
no pattern that distinguishes a judge's surname from any other English word,
and no pattern that distinguishes a Hindi/Sanskrit proper noun from a
Hindi/Sanskrit common word transliterated into English. Categorizing these
without a name list will silently produce garbage (either matching nothing,
so the category always reports 0 segments, or matching too much).

You already have the raw material to build these lists: the SC transcripts
have consistent ALL-CAPS speaker labels ("MR. CHANDRACHUD:", "HON'BLE MR.
JUSTICE HEGDE:", etc.). Run `extract_speaker_names()` below over your raw
transcript PDFs/text (before segmentation) to auto-build both lists, then
hand-review/dedupe once. This is a one-time ~10 minute pass, not a per-run
cost.

USAGE (from 06_evaluate.py)
----------------------------
    from category_eval import build_category_matcher, compute_per_category_metrics

    matcher = build_category_matcher(
        judge_advocate_names_file="./data/judge_advocate_names.txt",
        hindi_sanskrit_names_file="./data/hindi_sanskrit_names.txt",
    )

    # after you already have `preds, refs = transcribe_whisper(...)` etc.
    cat_metrics = compute_per_category_metrics(records, preds, refs, matcher)

    # cat_metrics looks like:
    # {
    #   "statute_citations":  {"WER (%)": 6.1, "CER (%)": ..., "Samples": 42},
    #   "case_numbers":       {...},
    #   ...
    # }
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

# jiwer is already a dependency of 06_evaluate.py.
# jiwer >=4.0 removed compute_measures() in favor of process_words();
# support both so this module works with whatever jiwer version your
# 06_evaluate.py environment has pinned.
try:
    from jiwer import compute_measures  # jiwer < 4.0

    def _measures(refs: list[str], preds: list[str]) -> dict[str, float]:
        m = compute_measures(refs, preds)
        return {"wer": m["wer"], "mer": m["mer"], "wil": m["wil"]}

except ImportError:  # jiwer >= 4.0
    from jiwer import process_words

    def _measures(refs: list[str], preds: list[str]) -> dict[str, float]:
        m = process_words(refs, preds)
        return {"wer": m.wer, "mer": m.mer, "wil": m.wil}


# ---------------------------------------------------------------------------
# Word-form numbers
# ---------------------------------------------------------------------------
# IMPORTANT: your reference transcripts spell numbers as words
# ("section six a", not "section 6a" — see your own Example 1 in the
# report template). A digit-only regex silently matches almost nothing
# in statute_citations and case_numbers. Cover both digit and word forms.

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
]
_COMPOUNDS = [f"{t}[\\s-]{o}" for t in _TENS for o in _ONES[1:10]]
_WORD_NUMBERS = sorted(_ONES + _TENS + _COMPOUNDS, key=len, reverse=True)
_WORD_NUM_ALT = "(?:" + "|".join(_WORD_NUMBERS) + ")"

# A statute-style number: digits ("6", "21") or spelled-out ("six",
# "twenty one"), optionally followed by a letter suffix ("a", "b") which
# itself may be spoken as a bare letter word.
_STATUTE_NUM = rf"(?:\d+[a-z]?|{_WORD_NUM_ALT}(?:\s+[a-z]\b)?)"

# A case-number style number sequence: "1234", "twelve thirty four", or a
# slash-separated year pair. Case numbers are less standardized in speech
# than statute sections, so this covers the digit form (most common in
# your dataset per Example testing) plus a basic word-number fallback.
_CASE_NUM = rf"(?:\d{{1,6}}|{_WORD_NUM_ALT}(?:\s+{_WORD_NUM_ALT})*)"


# ---------------------------------------------------------------------------
# Regex-matchable categories (no name list needed)
# ---------------------------------------------------------------------------

CATEGORY_PATTERNS: dict[str, re.Pattern] = {
    # "case numbers" checked before "statute citations" since e.g.
    # "SLP 1234/2023" could otherwise never fire — no shared tokens, but
    # ordering still matters for some overlapping edge cases, so keep this
    # priority order when adding patterns.
    "case_numbers": re.compile(
        rf"\b(slp|w\.?\s?p\.?|crl\.?\s?a\.?|c\.?\s?a\.?|civil\s+appeal|"
        rf"writ\s+petition|special\s+leave\s+petition|s\.?l\.?p\.?)\b"
        rf"|\b{_CASE_NUM}\s*(?:/|of)\s*\d{{4}}\b",
        re.IGNORECASE,
    ),
    "statute_citations": re.compile(
        rf"\b(section|article|clause|sub-?section|order|rule)\s+{_STATUTE_NUM}\b",
        re.IGNORECASE,
    ),
    "latin_legal_terms": re.compile(
        r"\b(suo\s*mot[uo]|locus\s*standi|prima\s*facie|ratio\s*decidendi|"
        r"ex\s*parte|inter\s*alia|mutatis\s*mutandis|ab\s*initio|"
        r"ipso\s*facto|sine\s*die|amicus\s*curiae|obiter\s*dict[au]m?|"
        r"res\s*judicata|bona\s*fide|per\s*incuriam|stare\s*decisis|"
        r"in\s*camera|de\s*novo|habeas\s*corpus)\b",
        re.IGNORECASE,
    ),
}

CATEGORY_ORDER = [
    "case_numbers",
    "statute_citations",
    "latin_legal_terms",
    "proper_nouns_judges_advocates",
    "hindi_sanskrit_proper_nouns",
    "general_legal_argument",  # fallback, must be last
]


# ---------------------------------------------------------------------------
# Name-list categories (need extraction from your transcripts)
# ---------------------------------------------------------------------------

def extract_speaker_names(raw_transcript_paths: list[str]) -> set[str]:
    """
    Pull candidate names from ALL-CAPS speaker labels in raw SC transcripts.

    Matches lines like:
        MR. CHANDRACHUD:
        HON'BLE MR. JUSTICE HEGDE:
        MS. SHYEL TREHAN:

    Returns a set of lowercase surnames/names, stripped of honorifics.
    This is a starting point — review the output once before using it as
    your ground-truth name list, since OCR noise or unusual formatting in
    a handful of transcripts can produce junk entries.
    """
    honorific_pattern = re.compile(
        r"^\s*(HON'?BLE\s+)?(MR\.?|MS\.?|MRS\.?|DR\.?|JUSTICE|CHIEF\s+JUSTICE)\s+",
        re.IGNORECASE,
    )
    label_pattern = re.compile(r"^\s*([A-Z][A-Z\.\s']{2,40}):\s")

    names: set[str] = set()
    for path in raw_transcript_paths:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            m = label_pattern.match(line)
            if not m:
                continue
            label = m.group(1)
            label = honorific_pattern.sub("", label).strip()
            label = re.sub(r"[.\s]+$", "", label)
            if 2 <= len(label.split()) <= 4 or len(label) >= 3:
                # keep individual name tokens, not the full label string,
                # since ASR errors usually hit a single surname
                for tok in label.split():
                    tok_clean = tok.strip(".").lower()
                    if len(tok_clean) >= 3:
                        names.add(tok_clean)
    return names


def load_name_list(path: str) -> set[str]:
    """One name per line, case-insensitive, '#' comments allowed."""
    names: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line.lower())
    return names


def build_category_matcher(
    judge_advocate_names_file: str = "",
    hindi_sanskrit_names_file: str = "",
) -> dict[str, re.Pattern | None]:
    """
    Build compiled word-boundary regexes from name-list files.

    If a file isn't provided, that category is disabled (utterances will
    fall through to general_legal_argument or another category instead of
    being silently mis-tagged).
    """
    matcher: dict[str, re.Pattern | None] = {
        "proper_nouns_judges_advocates": None,
        "hindi_sanskrit_proper_nouns": None,
    }

    if judge_advocate_names_file:
        names = load_name_list(judge_advocate_names_file)
        if names:
            pattern = r"\b(" + "|".join(re.escape(n) for n in sorted(names)) + r")\b"
            matcher["proper_nouns_judges_advocates"] = re.compile(pattern, re.IGNORECASE)

    if hindi_sanskrit_names_file:
        names = load_name_list(hindi_sanskrit_names_file)
        if names:
            pattern = r"\b(" + "|".join(re.escape(n) for n in sorted(names)) + r")\b"
            matcher["hindi_sanskrit_proper_nouns"] = re.compile(pattern, re.IGNORECASE)

    return matcher


# ---------------------------------------------------------------------------
# Categorization + per-category metrics
# ---------------------------------------------------------------------------

def categorize_utterance(
    text: str,
    matcher: dict[str, re.Pattern | None] | None = None,
) -> str:
    """
    Assign exactly one category per utterance, checked in CATEGORY_ORDER.
    First match wins — an utterance containing both a case number and a
    Latin term is tagged as case_numbers (adjust CATEGORY_ORDER if you'd
    rather prioritize differently).
    """
    matcher = matcher or {}

    if matcher.get("proper_nouns_judges_advocates") and \
            matcher["proper_nouns_judges_advocates"].search(text):
        return "proper_nouns_judges_advocates"
    if matcher.get("hindi_sanskrit_proper_nouns") and \
            matcher["hindi_sanskrit_proper_nouns"].search(text):
        return "hindi_sanskrit_proper_nouns"
    if CATEGORY_PATTERNS["case_numbers"].search(text):
        return "case_numbers"
    if CATEGORY_PATTERNS["statute_citations"].search(text):
        return "statute_citations"
    if CATEGORY_PATTERNS["latin_legal_terms"].search(text):
        return "latin_legal_terms"
    return "general_legal_argument"


def compute_per_category_metrics(
    records: list[dict],
    predictions: list[str],
    references: list[str],
    matcher: dict[str, re.Pattern | None] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Bucket (prediction, reference) pairs by category (matched against the
    REFERENCE text — you want ground truth to decide the bucket, not the
    model's possibly-wrong output) and compute WER/CER/MER/WIL per bucket.

    Categories with zero matched segments are omitted from the result
    rather than reported as 0.00% — a 0.00% WER on an empty bucket is a
    misleading number, not a real result.
    """
    buckets: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"refs": [], "preds": []}
    )

    for pred, ref in zip(predictions, references):
        cat = categorize_utterance(ref, matcher)
        buckets[cat]["refs"].append(ref)
        buckets[cat]["preds"].append(pred)

    results: dict[str, dict[str, Any]] = {}
    for cat in CATEGORY_ORDER:
        d = buckets.get(cat)
        if not d or not d["refs"]:
            continue
        word_m = _measures(d["refs"], d["preds"])
        char_refs = [" ".join(list(r)) for r in d["refs"]]
        char_preds = [" ".join(list(p)) for p in d["preds"]]
        char_m = _measures(char_refs, char_preds)
        results[cat] = {
            "WER (%)": round(word_m["wer"] * 100, 2),
            "CER (%)": round(char_m["wer"] * 100, 2),
            "MER (%)": round(word_m["mer"] * 100, 2),
            "WIL (%)": round(word_m["wil"] * 100, 2),
            "Samples": len(d["refs"]),
        }
    return results


def print_category_comparison(
    baseline_cat_metrics: dict[str, dict[str, Any]],
    finetuned_cat_metrics: dict[str, dict[str, Any]],
    baseline_label: str = "Baseline",
    finetuned_label: str = "Fine-tuned",
) -> None:
    """Markdown-ready table: category | baseline WER | fine-tuned WER | n."""
    print("\n| Utterance category | {b} WER | {f} WER | Δ (abs) | n (test set) |".format(
        b=baseline_label, f=finetuned_label
    ))
    print("|---|---|---|---|---|")
    all_cats = [c for c in CATEGORY_ORDER
                if c in baseline_cat_metrics or c in finetuned_cat_metrics]
    for cat in all_cats:
        b = baseline_cat_metrics.get(cat)
        f = finetuned_cat_metrics.get(cat)
        b_wer = f"{b['WER (%)']:.2f}%" if b else "—"
        f_wer = f"{f['WER (%)']:.2f}%" if f else "—"
        n = (f or b or {}).get("Samples", "—")
        delta = ""
        if b and f:
            d = f["WER (%)"] - b["WER (%)"]
            sign = "+" if d >= 0 else ""
            delta = f"{sign}{d:.2f}%"
        else:
            delta = "—"
        label = cat.replace("_", " ").title()
        print(f"| {label} | {b_wer} | {f_wer} | {delta} | {n} |")
    print()