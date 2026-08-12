#!/usr/bin/env python3
"""
Step 5 – Fine-tune a Whisper (or Distil-Whisper) model on the Supreme Court
         dataset using LoRA.

Supported models
----------------
  openai/whisper-large-v3          default; best accuracy
  openai/whisper-large-v3-turbo    ~4× faster decoding, slightly lower WER
  distil-whisper/distil-large-v3   6× faster inference, minimal WER loss
  openai/whisper-medium            lower VRAM (< 12 GB) option

Pass the desired model with:
    --model-id distil-whisper/distil-large-v3

Bug fixed vs previous version
------------------------------
Using task_type=TaskType.SEQ_2_SEQ_LM wraps the model in PEFT's
PeftModelForSeq2SeqLM, whose forward() explicitly passes input_ids=... to the
base model.  Whisper's forward() has no input_ids parameter (it uses
input_features), which causes:

  TypeError: WhisperForConditionalGeneration.forward() got an unexpected
             keyword argument 'input_ids'

Fix: omit task_type so PEFT uses the generic PeftModel passthrough wrapper.
Additional required changes:
  • model.config.use_cache = False      (mandatory for gradient checkpointing)
  • model.enable_input_require_grads()  (PEFT + gradient_checkpointing)
  • gradient_checkpointing_kwargs={"use_reentrant": False}  (PyTorch ≥ 2.1)

GPU requirements
----------------
  Model                    Min VRAM   Recommended
  whisper-large-v3 + LoRA  16 GB      24–80 GB
  distil-large-v3  + LoRA   8 GB      16 GB
  whisper-medium   + LoRA   8 GB      12 GB

Usage:
    python 05_train_whisper_lora.py \\
        --train-manifest ./data/manifests/train.jsonl \\
        --val-manifest   ./data/manifests/val.jsonl \\
        --output-dir     ./runs/whisper_lora

    # Distil-Whisper (faster):
    python 05_train_whisper_lora.py \\
        --model-id distil-whisper/distil-large-v3 \\
        --train-manifest ./data/manifests/train.jsonl \\
        --val-manifest   ./data/manifests/val.jsonl \\
        --output-dir     ./runs/distil_lora

    # Smoke test (5 training steps):
    python 05_train_whisper_lora.py \\
        --train-manifest ./data/manifests/train.jsonl \\
        --val-manifest   ./data/manifests/val.jsonl \\
        --output-dir     ./runs/smoke \\
        --max-steps 5 --batch-size 2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
import evaluate as hf_evaluate


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SupremeCourtDataset(Dataset):
    """
    PyTorch Dataset that reads a NeMo-format JSONL manifest.

    Each manifest line:  {"audio_filepath": "...", "text": "...", "duration": ...}

    Returns per item:
        input_features  – log-mel spectrogram (float32 numpy array)
        labels          – list of token IDs for the reference transcript
    """

    def __init__(
        self,
        manifest_path: str,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: WhisperTokenizer,
        max_samples: int = 0,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.records: list[dict] = []

        with open(manifest_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        if max_samples > 0:
            self.records = self.records[:max_samples]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]

        audio, _ = sf.read(rec["audio_filepath"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        input_features = self.feature_extractor(
            audio, sampling_rate=16_000, return_tensors="np"
        ).input_features[0]

        label_ids = self.tokenizer(rec["text"]).input_ids

        return {"input_features": input_features, "labels": label_ids}


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class WhisperCollator:
    """
    Pad a list of dataset items into a single batch tensor dict.

    The batch will contain exactly two keys:
      input_features  – padded log-mel spectrograms
      labels          – padded token IDs (-100 at padding positions)
    """
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_feats  = [{"input_features": f["input_features"]} for f in features]
        label_feats  = [{"input_ids":      f["labels"]}         for f in features]

        batch = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")

        labels_batch = self.processor.tokenizer.pad(label_feats, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Whisper tokeniser prepends BOS; remove it (decoder generates it)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# LoRA setup
# ---------------------------------------------------------------------------

def apply_lora(model: WhisperForConditionalGeneration, rank: int = 32):
    """
    Wrap Whisper with LoRA adapters on the Q and V projections of every
    attention layer in both encoder and decoder.

    Critical: do NOT set task_type=SEQ_2_SEQ_LM.
    That would make PEFT use PeftModelForSeq2SeqLM whose forward() explicitly
    passes input_ids=... to the base model, but Whisper.forward() has no
    input_ids parameter → TypeError at the very first training step.
    Omitting task_type uses the generic PeftModel which passes all kwargs
    through unchanged.
    """
    # Required before LoRA when using gradient checkpointing
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        # task_type intentionally omitted — see docstring above
        r              = rank,
        lora_alpha     = rank * 2,
        target_modules = ["q_proj", "v_proj"],
        lora_dropout   = 0.05,
        bias           = "none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Evaluation metric
# ---------------------------------------------------------------------------

def make_compute_metrics(processor: WhisperProcessor):
    wer_metric = hf_evaluate.load("wer")

    def compute_metrics(pred) -> dict[str, float]:
        pred_ids  = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        predictions = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        references  = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=predictions, references=references)
        return {"wer": round(wer, 2)}

    return compute_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fine-tune Whisper / Distil-Whisper with LoRA on Supreme Court audio."
    )
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--val-manifest",   required=True)
    ap.add_argument(
        "--model-id", default="openai/whisper-large-v3",
        help=(
            "HuggingFace model ID.  Options:\n"
            "  openai/whisper-large-v3          (default, best accuracy)\n"
            "  openai/whisper-large-v3-turbo    (4× faster decoding)\n"
            "  distil-whisper/distil-large-v3   (6× faster inference)\n"
            "  openai/whisper-medium            (low VRAM, < 12 GB)"
        ),
    )
    ap.add_argument("--output-dir",     default="./runs/whisper_lora")
    ap.add_argument("--batch-size",     type=int,   default=8)
    ap.add_argument("--grad-accum",     type=int,   default=2)
    ap.add_argument("--epochs",         type=int,   default=3)
    ap.add_argument("--max-steps",      type=int,   default=-1,
                    help="Override epochs (use for smoke tests, e.g. --max-steps 5).")
    ap.add_argument("--lr",             type=float, default=1e-4)
    ap.add_argument("--warmup-steps",   type=int,   default=500)
    ap.add_argument("--lora-rank",      type=int,   default=32,
                    help="LoRA rank r.  32 is a good default; try 64 on H100/A100.")
    ap.add_argument("--fp16",           type=lambda x: x.lower() != "false", default=True)
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-val-samples",   type=int, default=0)
    ap.add_argument("--resume-from",    default="")
    args = ap.parse_args()

    use_gpu  = torch.cuda.is_available()
    use_fp16 = args.fp16 and use_gpu
    if args.fp16 and not use_gpu:
        print("WARNING: fp16 requested but no GPU found → falling back to fp32.")

    gpu_name = torch.cuda.get_device_name(0) if use_gpu else "CPU"
    print(f"Model  : {args.model_id}")
    print(f"Device : {gpu_name}")
    print(f"FP16   : {use_fp16}")

    # ── Load processor + model ────────────────────────────────────────────
    processor = WhisperProcessor.from_pretrained(
        args.model_id, language="English", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    model.generation_config.language         = "English"
    model.generation_config.task             = "transcribe"
    model.generation_config.forced_decoder_ids = None

    model = apply_lora(model, rank=args.lora_rank)

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = SupremeCourtDataset(
        args.train_manifest, processor.feature_extractor, processor.tokenizer,
        max_samples=args.max_train_samples,
    )
    val_ds = SupremeCourtDataset(
        args.val_manifest, processor.feature_extractor, processor.tokenizer,
        max_samples=args.max_val_samples,
    )
    print(f"Train  : {len(train_ds):,} segments")
    print(f"Val    : {len(val_ds):,} segments")

    # ── Training arguments ────────────────────────────────────────────────
    common = dict(
        output_dir                      = args.output_dir,
        per_device_train_batch_size     = args.batch_size,
        per_device_eval_batch_size      = args.batch_size,
        gradient_accumulation_steps     = args.grad_accum,
        # gradient_checkpointing requires use_reentrant=False with PEFT
        gradient_checkpointing          = True,
        gradient_checkpointing_kwargs   = {"use_reentrant": False},
        learning_rate                   = args.lr,
        warmup_steps                    = args.warmup_steps,
        num_train_epochs                = args.epochs,
        max_steps                       = args.max_steps,
        fp16                            = use_fp16,
        predict_with_generate           = True,
        generation_max_length           = 225,
        save_strategy                   = "epoch",
        logging_steps                   = 50,
        report_to                       = "none",
        load_best_model_at_end          = True,
        metric_for_best_model           = "wer",
        greater_is_better               = False,
        dataloader_num_workers          = 4,
        remove_unused_columns           = False,
    )

    # Gracefully handle transformers versions that use evaluation_strategy
    try:
        training_args = Seq2SeqTrainingArguments(**common, eval_strategy="epoch")
    except TypeError:
        # gradient_checkpointing_kwargs may also be unsupported in older versions
        common.pop("gradient_checkpointing_kwargs", None)
        try:
            training_args = Seq2SeqTrainingArguments(**common, eval_strategy="epoch")
        except TypeError:
            training_args = Seq2SeqTrainingArguments(**common, evaluation_strategy="epoch")

    # ── Trainer ───────────────────────────────────────────────────────────
    collator        = WhisperCollator(processor=processor)
    compute_metrics = make_compute_metrics(processor)

    trainer_kwargs = dict(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collator,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )
    # Gracefully handle old transformers that use 'tokenizer' kwarg
    try:
        trainer = Seq2SeqTrainer(**trainer_kwargs, processing_class=processor.tokenizer)
    except TypeError:
        trainer = Seq2SeqTrainer(**trainer_kwargs, tokenizer=processor.feature_extractor)

    # ── Train ─────────────────────────────────────────────────────────────
    trainer.train(resume_from_checkpoint=args.resume_from or None)

    # ── Save ──────────────────────────────────────────────────────────────
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"\nModel saved → {final_dir}")
    print("Run 06_evaluate.py to compute WER on the test set.")


if __name__ == "__main__":
    main()
