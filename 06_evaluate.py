#!/usr/bin/env python3
"""
Step 6 – Evaluate ASR models on the test manifest with a multi-baseline
         comparison table.

Supports two model families:
  • Whisper (HuggingFace)  – zero-shot baseline and/or Whisper+LoRA fine-tuned
  • NeMo                   – zero-shot baselines and/or NeMo fine-tuned (.nemo)

Metrics computed
----------------
  WER   Word Error Rate          – primary ASR metric (lower is better)
  CER   Character Error Rate     – complements WER for vocabulary-heavy text
  MER   Match Error Rate         – accounts for insertion/deletion asymmetry
  WIL   Word Information Lost    – information-theoretic view of errors

Usage examples
--------------
    # Fine-tuned NeMo model vs two zero-shot NeMo baselines:
    python 06_evaluate.py \\
        --manifest        ./data/manifests/test.jsonl \\
        --nemo-model      ./runs/nemo_hybrid/final.nemo \\
        --nemo-baselines  stt_en_conformer_ctc_large,stt_en_conformer_transducer_large \\
        --output-json     ./runs/eval_results.json

    # Fine-tuned Whisper LoRA vs zero-shot Whisper:
    python 06_evaluate.py \\
        --manifest      ./data/manifests/test.jsonl \\
        --whisper-model ./runs/whisper_lora/final \\
        --whisper-baselines openai/whisper-large-v3 \\
        --output-json   ./runs/eval_results.json

    # Full comparison — all four models:
    python 06_evaluate.py \\
        --manifest           ./data/manifests/test.jsonl \\
        --nemo-model         ./runs/nemo_hybrid/final.nemo \\
        --nemo-baselines     stt_en_conformer_ctc_large,stt_en_conformer_transducer_large \\
        --whisper-baselines  openai/whisper-large-v3 \\
        --output-json        ./runs/eval_results.json

    # Quick sanity check on 50 segments:
    python 06_evaluate.py \\
        --manifest      ./data/manifests/test.jsonl \\
        --nemo-model    ./runs/nemo_hybrid/final.nemo \\
        --max-samples   50
"""
  # empty = only the 4 pattern-based categories
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from jiwer import compute_measures
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(path: str, max_samples: int = 0) -> list[dict]:
    """Read a NeMo JSONL manifest. Returns list of {audio_filepath, text} dicts."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]
    return records


# ---------------------------------------------------------------------------
# Whisper model loading and transcription
# ---------------------------------------------------------------------------
# We deliberately avoid transformers.pipeline() because newer versions try to
# import torchcodec (which needs FFmpeg) even when we supply raw numpy audio.
# Instead we call WhisperForConditionalGeneration.generate() directly, loading
# audio ourselves with soundfile (already installed).
# ---------------------------------------------------------------------------

def load_whisper_pipeline(model_path: str, device: str):
    """
    Load a Whisper model + processor.

    Returns a dict  {"model": ..., "processor": ..., "device": ..., "dtype": ...}
    so transcribe_whisper can call model.generate() directly without pipeline().

    Handles:
      • plain pretrained models (e.g. openai/whisper-large-v3)
      • full fine-tuned directories (saved with trainer.save_model())
      • LoRA fine-tuned directories (detected by adapter_config.json)
    """
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError:
        raise SystemExit(
            "transformers is required for Whisper inference.\n"
            "Install: pip install transformers"
        )

    dtype      = torch.float16 if device == "cuda" else torch.float32
    torch_dev  = torch.device(device)

    # ── LoRA adapter: merge weights before loading ─────────────────────────
    adapter_cfg = Path(model_path) / "adapter_config.json"
    if adapter_cfg.exists():
        try:
            from peft import PeftModel
        except ImportError:
            raise SystemExit("peft is required for LoRA models: pip install peft")
        cfg     = json.loads(adapter_cfg.read_text())
        base_id = cfg.get("base_model_name_or_path", "openai/whisper-large-v3")
        print(f"    LoRA adapter detected. Loading base: {base_id}")
        base      = WhisperForConditionalGeneration.from_pretrained(base_id, torch_dtype=dtype)
        peft_model = PeftModel.from_pretrained(base, model_path)
        model      = peft_model.merge_and_unload()
        processor  = WhisperProcessor.from_pretrained(model_path)
    else:
        model     = WhisperForConditionalGeneration.from_pretrained(
                        model_path, torch_dtype=dtype)
        # Processor may be saved alongside the model (fine-tuned) or fetched
        # from the Hub (pretrained baseline).
        try:
            processor = WhisperProcessor.from_pretrained(model_path)
        except Exception:
            processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

    model = model.to(torch_dev)
    model.eval()

    return {"model": model, "processor": processor,
            "device": torch_dev, "dtype": dtype}


def transcribe_whisper(
    whisper_bundle: dict,
    records: list[dict],
    batch_size: int = 8,
) -> tuple[list[str], list[str]]:
    """
    Transcribe audio files using model.generate() directly.
    Avoids transformers.pipeline() and its torchcodec/FFmpeg dependency.
    """
    model     = whisper_bundle["model"]
    processor = whisper_bundle["processor"]
    device    = whisper_bundle["device"]
    dtype     = whisper_bundle["dtype"]

    predictions: list[str] = []
    references:  list[str] = []

    for i in tqdm(range(0, len(records), batch_size),
                  desc="    Transcribing (Whisper)", leave=False):
        batch_recs = records[i : i + batch_size]

        # Load audio for each record in the batch
        audios = []
        for rec in batch_recs:
            audio, sr = sf.read(rec["audio_filepath"], dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # Resample if needed (segments should already be 16 kHz)
            if sr != 16000:
                audio = _linear_resample(audio, sr, 16000)
            audios.append(audio)
            references.append(rec["text"].strip().lower())

        # Feature extraction (log-mel spectrogram)
        inputs = processor.feature_extractor(
            audios,
            sampling_rate = 16000,
            return_tensors = "pt",
            padding        = True,
        )
        input_features = inputs.input_features.to(device=device, dtype=dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                language = "english",
                task     = "transcribe",
            )

        batch_preds = processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        predictions.extend([p.strip().lower() for p in batch_preds])

    return predictions, references


def _linear_resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear-interpolation resample (no librosa/ffmpeg needed)."""
    new_len = int(len(audio) * target_sr / orig_sr)
    return np.interp(
        np.linspace(0, len(audio) - 1, new_len),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# NeMo model loading and transcription
# ---------------------------------------------------------------------------

def load_nemo_model(model_path: str):
    """
    Load a NeMo ASR model from either:
      • A .nemo checkpoint file (fine-tuned)
      • A NGC pretrained model name (zero-shot baseline)

    Auto-detects which ASR model class to use via NeMo's ASRModel.restore_from
    (which infers the class from the checkpoint metadata).
    """
    try:
        import nemo.collections.asr as nemo_asr  # noqa: PLC0415
    except ImportError:
        raise SystemExit(
            "nemo_toolkit[asr] is required for NeMo inference.\n"
            "Install: pip install nemo_toolkit[asr]"
        )

    if model_path.endswith(".nemo") and Path(model_path).exists():
        print(f"    Restoring NeMo checkpoint: {model_path}")
        model = nemo_asr.models.ASRModel.restore_from(model_path)
    else:
        # Treat as a pretrained NGC model name
        print(f"    Loading NeMo pretrained model: {model_path}")
        model = nemo_asr.models.ASRModel.from_pretrained(model_path)

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    return model


def _nemo_transcribe(model, audio_files: list[str], batch_size: int) -> list:
    """
    Call model.transcribe() using the correct keyword argument for this model type.

    NeMo's transcribe() API is inconsistent across model families:
      EncDecCTCModelBPE   → paths2audio_files=...
      EncDecRNNTBPEModel  → audio=...            (NeMo 1.18+)
      EncDecHybridRNNTCTCBPEModel → audio=...

    We inspect the signature at runtime to pick the right name, then fall back
    to the other if the first raises TypeError.
    """
    import inspect
    sig = inspect.signature(model.transcribe)
    params = set(sig.parameters)

    def _call(kwarg_name: str):
        return model.transcribe(**{kwarg_name: audio_files, "batch_size": batch_size})

    if "audio" in params:
        return _call("audio")
    if "paths2audio_files" in params:
        return _call("paths2audio_files")

    # Unknown signature — try both and raise the second error if both fail
    try:
        return _call("audio")
    except TypeError:
        return _call("paths2audio_files")


def transcribe_nemo(
    model,
    records: list[dict],
    batch_size: int = 16,
) -> tuple[list[str], list[str]]:
    """
    Transcribe using a NeMo ASRModel. Returns (preds, refs).

    NeMo's transcribe() handles batching internally and returns List[str].
    Hybrid models return a tuple (rnnt_hypotheses, ctc_hypotheses); we use
    the RNN-T (first) output.
    """
    audio_files = [rec["audio_filepath"] for rec in records]
    references  = [rec["text"].strip().lower() for rec in records]

    with torch.no_grad():
        raw_hypotheses = _nemo_transcribe(model, audio_files, batch_size)

    # Hybrid models return (rnnt_hyps, ctc_hyps) — take RNN-T output
    if isinstance(raw_hypotheses, tuple):
        raw_hypotheses = raw_hypotheses[0]

    predictions = [
        (h.text if hasattr(h, "text") else str(h)).strip().lower()
        for h in raw_hypotheses
    ]

    return predictions, references


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    predictions: list[str],
    references:  list[str],
) -> dict[str, Any]:
    """
    Compute WER, CER, MER, WIL using jiwer.

    CER is computed by treating each character as a "word".
    """
    word_measures = compute_measures(references, predictions)

    char_refs  = [" ".join(list(r)) for r in references]
    char_preds = [" ".join(list(p)) for p in predictions]
    char_measures = compute_measures(char_refs, char_preds)

    return {
        "WER (%)":       round(word_measures["wer"]  * 100, 2),
        "CER (%)":       round(char_measures["wer"]  * 100, 2),
        "MER (%)":       round(word_measures["mer"]  * 100, 2),
        "WIL (%)":       round(word_measures["wil"]  * 100, 2),
        "Insertions":    word_measures["insertions"],
        "Deletions":     word_measures["deletions"],
        "Substitutions": word_measures["substitutions"],
        "Samples":       len(predictions),
    }


# ---------------------------------------------------------------------------
# Pretty-print: per-model result and final comparison table
# ---------------------------------------------------------------------------

def print_model_metrics(label: str, metrics: dict) -> None:
    w = 46
    print(f"\n{'─' * w}")
    print(f"  {label}")
    print(f"{'─' * w}")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}")
    print(f"{'─' * w}")


def print_comparison_table(results: list[dict]) -> None:
    """
    Print a Markdown-compatible comparison table of all evaluated models.

    Each entry in `results` is:
        {"label": str, "metrics": dict, "is_finetuned": bool}
    """
    if not results:
        return

    print("\n")
    print("=" * 72)
    print("  BENCHMARK COMPARISON TABLE")
    print("=" * 72)

    col_w = 36
    header = f"{'Model':<{col_w}} {'WER':>7} {'CER':>7} {'MER':>7} {'WIL':>7}"
    sep    = f"{'─' * col_w} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 7}"
    print(header)
    print(sep)

    best_wer   = min(r["metrics"]["WER (%)"] for r in results)
    baseline_wers: list[float] = [
        r["metrics"]["WER (%)"] for r in results if not r.get("is_finetuned", False)
    ]

    for entry in results:
        m     = entry["metrics"]
        label = entry["label"]
        if len(label) > col_w - 1:
            label = label[: col_w - 4] + "..."

        wer = m["WER (%)"]
        marker = " *" if wer == best_wer else "  "
        print(
            f"{label:<{col_w}}{marker}"
            f"{wer:>6.2f}%"
            f"{m['CER (%)']:>7.2f}%"
            f"{m['MER (%)']:>7.2f}%"
            f"{m['WIL (%)']:>7.2f}%"
        )

    print(sep)
    print("  * = best WER\n")

    # WER improvement summary (fine-tuned vs each baseline)
    finetuned = [r for r in results if r.get("is_finetuned", False)]
    baselines = [r for r in results if not r.get("is_finetuned", False)]
    if finetuned and baselines:
        print("  WER Improvement (fine-tuned vs baselines):")
        for ft in finetuned:
            for bl in baselines:
                wer_ft = ft["metrics"]["WER (%)"]
                wer_bl = bl["metrics"]["WER (%)"]
                abs_imp = wer_bl - wer_ft
                rel_imp = 100 * abs_imp / wer_bl if wer_bl > 0 else 0.0
                sign = "+" if abs_imp >= 0 else ""
                print(
                    f"    {ft['label'][:28]} vs {bl['label'][:20]}: "
                    f"{sign}{abs_imp:.2f}% abs  ({rel_imp:.1f}% rel)"
                )
        print()

    print("=" * 72)


# ---------------------------------------------------------------------------
# Sample predictions
# ---------------------------------------------------------------------------

def print_sample_predictions(
    label: str,
    refs: list[str],
    preds: list[str],
    n: int = 5,
) -> None:
    print(f"\n  Sample predictions — {label} (first {n}):")
    for ref, pred in zip(refs[:n], preds[:n]):
        print(f"    REF : {ref}")
        print(f"    PRED: {pred}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate ASR models on the Supreme Court test set.\n"
            "Supports Whisper (HuggingFace) and NeMo models with a "
            "multi-baseline comparison table."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--manifest",   required=True,
                    help="Path to test.jsonl manifest.")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Limit to first N segments (0 = all).")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu",
                    choices=["cuda", "cpu"])
    ap.add_argument("--output-json", default="",
                    help="Save all results to this JSON file.")

    # ── NeMo args ─────────────────────────────────────────────────────────
    ap.add_argument(
        "--nemo-model", default="",
        help=(
            "Path to the fine-tuned NeMo .nemo checkpoint.\n"
            "Example: ./runs/nemo_hybrid/final.nemo"
        ),
    )
    ap.add_argument(
        "--nemo-baselines", default="",
        help=(
            "Comma-separated NeMo pretrained model names to use as zero-shot\n"
            "baselines.  Example:\n"
            "  stt_en_conformer_ctc_large,stt_en_conformer_transducer_large"
        ),
    )
    ap.add_argument("--nemo-batch-size", type=int, default=16,
                    help="Batch size for NeMo transcription.")

    # ── Whisper args ───────────────────────────────────────────────────────
    ap.add_argument(
        "--whisper-model", default="",
        help=(
            "Path to the fine-tuned Whisper model directory (plain or LoRA).\n"
            "Example: ./runs/whisper_lora/final"
        ),
    )
    ap.add_argument(
        "--whisper-baselines", default="",
        help=(
            "Comma-separated HuggingFace Whisper model IDs to use as zero-shot\n"
            "baselines.  Example: openai/whisper-large-v3"
        ),
    )
    args = ap.parse_args()

    # ── Load test set ─────────────────────────────────────────────────────
    records = load_manifest(args.manifest, max_samples=args.max_samples)
    if not records:
        raise ValueError(f"No records found in manifest: {args.manifest}")
    print(f"\nEvaluating on {len(records)} segments from {args.manifest}")
    print(f"Device: {args.device}\n")

    all_results: list[dict] = []

    # ── NeMo zero-shot baselines ───────────────────────────────────────────
    if args.nemo_baselines:
        for name in [n.strip() for n in args.nemo_baselines.split(",") if n.strip()]:
            label = f"NeMo (zero-shot): {name}"
            print(f"[Baseline] {label}")
            model = load_nemo_model(name)
            preds, refs = transcribe_nemo(model, records, batch_size=args.nemo_batch_size)
            metrics = compute_all_metrics(preds, refs)
            print_model_metrics(label, metrics)
            print_sample_predictions(label, refs, preds)
            all_results.append({
                "label":       label,
                "metrics":     metrics,
                "is_finetuned": False,
                "model_id":    name,
                "framework":   "nemo",
            })
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Whisper zero-shot baselines ────────────────────────────────────────
    if args.whisper_baselines:
        for name in [n.strip() for n in args.whisper_baselines.split(",") if n.strip()]:
            label = f"Whisper (zero-shot): {name}"
            print(f"[Baseline] {label}")
            pipe   = load_whisper_pipeline(name, args.device)
            preds, refs = transcribe_whisper(pipe, records)
            metrics = compute_all_metrics(preds, refs)
            print_model_metrics(label, metrics)
            print_sample_predictions(label, refs, preds)
            all_results.append({
                "label":       label,
                "metrics":     metrics,
                "is_finetuned": False,
                "model_id":    name,
                "framework":   "whisper",
            })
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Fine-tuned NeMo model ─────────────────────────────────────────────
    if args.nemo_model:
        label = f"NeMo fine-tuned: {Path(args.nemo_model).name}"
        print(f"[Fine-tuned] {label}")
        model = load_nemo_model(args.nemo_model)
        preds, refs = transcribe_nemo(model, records, batch_size=args.nemo_batch_size)
        metrics = compute_all_metrics(preds, refs)
        print_model_metrics(label, metrics)
        print_sample_predictions(label, refs, preds)
        all_results.append({
            "label":       label,
            "metrics":     metrics,
            "is_finetuned": True,
            "model_id":    args.nemo_model,
            "framework":   "nemo",
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Fine-tuned Whisper model ───────────────────────────────────────────
    if args.whisper_model:
        label = f"Whisper fine-tuned: {Path(args.whisper_model).name}"
        print(f"[Fine-tuned] {label}")
        pipe   = load_whisper_pipeline(args.whisper_model, args.device)
        preds, refs = transcribe_whisper(pipe, records)
        metrics = compute_all_metrics(preds, refs)
        print_model_metrics(label, metrics)
        print_sample_predictions(label, refs, preds)
        all_results.append({
            "label":       label,
            "metrics":     metrics,
            "is_finetuned": True,
            "model_id":    args.whisper_model,
            "framework":   "whisper",
        })

    if not all_results:
        print(
            "Nothing to evaluate. Provide at least one of:\n"
            "  --nemo-model, --nemo-baselines, "
            "--whisper-model, --whisper-baselines"
        )
        return

    # ── Comparison table ──────────────────────────────────────────────────
    print_comparison_table(all_results)

    # ── Save results ──────────────────────────────────────────────────────
    if args.output_json:
        output = {
            "manifest":    args.manifest,
            "n_samples":   len(records),
            "device":      args.device,
            "results":     all_results,
        }
        Path(args.output_json).write_text(
            json.dumps(output, indent=2), encoding="utf-8"
        )
        print(f"Results saved → {args.output_json}")


if __name__ == "__main__":
    main()
