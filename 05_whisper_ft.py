#!/usr/bin/env python3
"""
Step 5 – Fine-tune Whisper large-v3 on Supreme Court ASR data.

Strategy: freeze encoder, fine-tune decoder only.

  Whisper architecture
  ────────────────────
  Encoder  (CNN + 32 Transformer blocks)  ← FROZEN
      Converts mel spectrogram → acoustic embeddings.
      Already handles Indian English well.

  Decoder  (32 Transformer blocks, cross-attention + self-attention)
      Attends to encoder output to generate tokens.  ← TRAINED
      This is where domain vocabulary lives.

Literature basis:
  • "Keyword-Guided Adaptation of ASR" (Interspeech 2024):
    freeze encoder + fine-tune decoder = best for jargon boosting.
  • "Whisper" (Radford et al., 2022): decoder handles language/vocabulary,
    encoder handles acoustics — making them independently adaptable.

Dependencies (install once):
    pip install transformers>=4.40 accelerate>=0.30 soundfile jiwer evaluate

Usage:
    # decoder-only fine-tuning (recommended)
    python 05_whisper_ft.py

    # full fine-tuning (slower, use if accent is also a problem)
    python 05_whisper_ft.py --no-freeze-encoder

    # smoke test
    python 05_whisper_ft.py --max-steps 5 --eval-steps 5
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

try:
    import evaluate
    _wer_metric = evaluate.load("wer")
except Exception:
    _wer_metric = None


# =============================================================================
# Dataset
# =============================================================================

class SCManifestDataset(Dataset):
    """
    Reads a NeMo-format JSONL manifest and serves (input_features, labels).
    Each line: {"audio_filepath": ..., "text": ..., "duration": ...}
    """

    def __init__(
        self,
        manifest_path: str,
        processor: WhisperProcessor,
        max_duration_s: float = 29.5,
    ):
        self.processor = processor
        self.records: list[dict] = []

        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("duration", 0) <= max_duration_s:
                    self.records.append(rec)

        print(f"Loaded {len(self.records)} segments from {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]

        # Load audio
        audio, sr = sf.read(rec["audio_filepath"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # stereo → mono
        if sr != 16000:
            audio = _resample(audio, sr, 16000)

        # Extract log-mel features  (shape: [80, 3000])
        features = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features[0]

        # Tokenise reference text
        labels = self.processor.tokenizer(rec["text"]).input_ids

        return {"input_features": features, "labels": labels}


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear-interpolation resample (avoids librosa dependency)."""
    ratio = target_sr / orig_sr
    new_len = int(len(audio) * ratio)
    return np.interp(
        np.linspace(0, len(audio) - 1, new_len),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


# =============================================================================
# Data collator
# =============================================================================

@dataclass
class WhisperDataCollator:
    """
    Pad input features and labels to the longest item in the batch.
    Replaces padding token ids in labels with -100 so the loss ignores them.
    """
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # ── input features (all same size after feature extraction: [80, 3000]) ─
        input_batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt",
        )

        # ── labels (variable length: tokenised text) ──────────────────────────
        label_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            return_tensors="pt",
        )
        labels = label_batch["input_ids"].masked_fill(
            label_batch["attention_mask"].ne(1), -100
        )
        # Strip leading decoder-start token (Trainer prepends it during generation)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        input_batch["labels"] = labels
        return input_batch


# =============================================================================
# WER metric
# =============================================================================

def make_compute_metrics(processor: WhisperProcessor):
    def compute_metrics(pred):
        pred_ids   = pred.predictions
        label_ids  = pred.label_ids

        # Replace -100 padding with pad token id before decoding
        label_ids  = np.where(label_ids == -100,
                               processor.tokenizer.pad_token_id, label_ids)

        pred_str   = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str  = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Normalise whitespace
        pred_str  = [" ".join(s.split()) for s in pred_str]
        label_str = [" ".join(s.split()) for s in label_str]

        if _wer_metric is not None:
            wer = _wer_metric.compute(predictions=pred_str, references=label_str)
            return {"wer": round(wer, 4)}

        # Fallback: simple token-level WER if evaluate not available
        total_words = total_errors = 0
        for p, r in zip(pred_str, label_str):
            ref = r.split(); hyp = p.split()
            total_words  += len(ref)
            total_errors += abs(len(ref) - len(hyp))  # rough estimate
        return {"wer": round(total_errors / max(total_words, 1), 4)}

    return compute_metrics


# =============================================================================
# Freeze helpers
# =============================================================================

def _count_params(module: torch.nn.Module) -> tuple[int, int]:
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


def print_trainable_summary(model: torch.nn.Module) -> None:
    components = {
        "encoder (CNN + Transformer)": model.model.encoder,
        "decoder (cross-attn + self-attn)": model.model.decoder,
        "lm_head (output projection)": model.proj_out,
    }
    print("\n" + "─" * 66)
    print(f"  {'Component':<36} {'Trainable':>12} {'Total':>12}")
    print("─" * 66)
    for name, mod in components.items():
        tr, tot = _count_params(mod)
        tag = "  [FROZEN]" if tr == 0 else ""
        print(f"  {name:<36} {tr:>12,} {tot:>12,}{tag}")
    tr_all, tot_all = _count_params(model)
    print("─" * 66)
    print(f"  {'TOTAL':<36} {tr_all:>12,} {tot_all:>12,}")
    print("─" * 66 + "\n")


def freeze_encoder(model: WhisperForConditionalGeneration) -> None:
    """Freeze all encoder parameters (CNN + Transformer blocks)."""
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    print("[freeze] Encoder frozen — training decoder only.")
    print_trainable_summary(model)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--train-manifest",
                        default="./data/manifests/train.jsonl")
    parser.add_argument("--val-manifest",
                        default="./data/manifests/val.jsonl")
    parser.add_argument("--output-dir",
                        default="./runs/whisper_decoder_ft")

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument("--model-name",
                        default="openai/whisper-large-v3",
                        help="HuggingFace model id or local path.")
    parser.add_argument("--language", default="en",
                        help="Target language code (default: en).")
    parser.add_argument("--task", default="transcribe",
                        choices=["transcribe", "translate"])

    # ── Freeze strategy ───────────────────────────────────────────────────────
    parser.add_argument("--no-freeze-encoder", action="store_true",
                        default=False,
                        help="Unfreeze encoder for full fine-tuning.\n"
                             "Default: encoder frozen, decoder only.")

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--batch-size",   type=int,   default=4,
                        help="Per-device batch size.")
    parser.add_argument("--grad-accum",   type=int,   default=8,
                        help="Gradient accumulation steps.\n"
                             "Effective batch = batch-size × grad-accum.")
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int,   default=500)
    parser.add_argument("--max-steps",    type=int,   default=-1,
                        help="Override epochs (smoke test: --max-steps 5).")
    parser.add_argument("--eval-steps",   type=int,   default=500,
                        help="Run evaluation every N steps.")
    parser.add_argument("--save-steps",   type=int,   default=500)
    parser.add_argument("--patience",     type=int,   default=3,
                        help="Early stopping patience (eval rounds).")
    parser.add_argument("--num-workers",  type=int,   default=2)
    parser.add_argument("--fp16",         action="store_true", default=False)
    parser.add_argument("--bf16",         action="store_true", default=False)

    args = parser.parse_args()

    # ── Sanity checks ─────────────────────────────────────────────────────────
    for f in [args.train_manifest, args.val_manifest]:
        if not Path(f).exists():
            raise FileNotFoundError(
                f"Manifest not found: {f}\n"
                "Run 04_make_manifest.py first."
            )

    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        print("WARNING: No GPU detected — training will be very slow.")

    # Auto-detect precision
    if not args.fp16 and not args.bf16 and use_gpu:
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            args.bf16 = True   # Ampere+ → BF16
        else:
            args.fp16 = True   # Older GPUs → FP16

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    freeze_desc = (
        "full fine-tuning (encoder + decoder)"
        if args.no_freeze_encoder
        else "decoder-only (encoder frozen)"
    )
    eff_batch = args.batch_size * args.grad_accum

    print("=" * 66)
    print("  SC ASR — Whisper Fine-tuning")
    print(f"  Model        : {args.model_name}")
    print(f"  Freeze mode  : {freeze_desc}")
    print(f"  Device       : {torch.cuda.get_device_name(0) if use_gpu else 'CPU'}")
    print(f"  Precision    : {'bf16' if args.bf16 else 'fp16' if args.fp16 else 'fp32'}")
    print(f"  Batch        : {args.batch_size} × grad_accum {args.grad_accum} = {eff_batch}")
    print(f"  LR           : {args.lr}")
    print(f"  Output       : {out_dir}")
    print("=" * 66)

    # ── Processor & model ─────────────────────────────────────────────────────
    print(f"\nLoading processor and model: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        language=args.language,
        task=args.task,
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.config.forced_decoder_ids = None    # let the model generate freely
    model.config.suppress_tokens    = []

    # ── Freeze strategy ───────────────────────────────────────────────────────
    if not args.no_freeze_encoder:
        freeze_encoder(model)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = SCManifestDataset(args.train_manifest, processor)
    val_dataset   = SCManifestDataset(args.val_manifest,   processor)

    data_collator = WhisperDataCollator(
        processor               = processor,
        decoder_start_token_id  = model.config.decoder_start_token_id,
    )

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir                  = str(out_dir),
        num_train_epochs            = args.epochs,
        max_steps                   = args.max_steps,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        warmup_steps                = args.warmup_steps,
        lr_scheduler_type           = "linear",
        fp16                        = args.fp16,
        bf16                        = args.bf16,
        predict_with_generate       = True,
        generation_max_length       = 225,
        eval_strategy               = "steps",
        eval_steps                  = args.eval_steps,
        save_strategy               = "steps",
        save_steps                  = args.save_steps,
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,
        logging_steps               = 25,
        report_to                   = "none",       # disable wandb/tensorboard
        dataloader_num_workers      = args.num_workers,
        remove_unused_columns       = False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        data_collator   = data_collator,
        compute_metrics = make_compute_metrics(processor),
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=args.patience)],
        tokenizer       = processor.feature_extractor,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    final_dir = out_dir / "final_model"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    print(f"\nSaved model + processor to: {final_dir}")
    print(f"Next step:")
    print(f"  python 06_evaluate.py --whisper-model {final_dir}")


if __name__ == "__main__":
    main()
