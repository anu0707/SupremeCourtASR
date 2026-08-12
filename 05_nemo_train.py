#!/usr/bin/env python3
"""
Step 5 – Fine-tune a NeMo Conformer ASR model on Supreme Court hearings.

Compatible with:
    nemo-toolkit >= 2.7.3
    lightning     >= 2.4.0

Architectures:
    ctc    -> EncDecCTCModelBPE
    rnnt   -> EncDecRNNTBPEModel
    hybrid -> EncDecHybridRNNTCTCBPEModel

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fine-tuning strategy for domain / keyword adaptation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RNN-T has three trainable components:

  Encoder (Conformer) — 18 Conformer blocks, ~86M params
      Converts mel spectrogram -> acoustic embeddings.
      Learns *how audio sounds*: phonemes, accent, noise.

  Prediction Network (decoder) — LSTM over token history, ~5M params
      Acts as an implicit language model inside RNN-T.
      Learns *what token comes next* given transcription history.

  Joint Network — small FF that fuses encoder + decoder, <1M params
      Learns *when* to emit a token.

Literature-backed strategy for this use-case
─────────────────────────────────────────────
• "Fast Text-Only Domain Adaptation of RNN-T Prediction Network"
  (Arxiv 2104.11127): updating only the prediction network gives
  10-45% relative WER reduction on domain-shifted test sets.
  The prediction network is the language-model half of RNN-T.

• "Keyword-Guided Adaptation of ASR" (Interspeech 2024):
  freeze encoder + fine-tune decoder is the best strategy for
  jargon/keyword boosting. Prompt-tuning can work with ~15K params.

• Supreme Court recordings: primary challenge is *vocabulary*
  (legal terms, Indian names, case citations like "SLP 1234/2023"),
  not acoustics — the base Conformer already handles Indian English.

Recommended workflow:
  Stage 1  --decoder-only              (fast, 5M params, safe)
  Stage 2  --unfreeze-top-n-layers 4   (if WER plateaus; tune accent)
  Stage 3  --unfreeze-epoch 1          (full FT warm-up; last resort)

Usage examples:
  # Stage 1 – decoder only
  python 05_nemo_train.py --train-manifest data/train.json \\
      --val-manifest data/val.json --decoder-only

  # Stage 2 – top 4 encoder layers + decoder
  python 05_nemo_train.py --train-manifest data/train.json \\
      --val-manifest data/val.json --unfreeze-top-n-layers 4

  # Stage 3 – warm-up (freeze encoder epoch 0, unfreeze epoch 1)
  python 05_nemo_train.py --train-manifest data/train.json \\
      --val-manifest data/val.json --unfreeze-epoch 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import open_dict


# =============================================================================
# Lazy NeMo import  (avoids slow import at module level)
# =============================================================================

_nemo_asr = None


def get_nemo_asr():
    global _nemo_asr
    if _nemo_asr is None:
        import nemo.collections.asr as nemo_asr
        _nemo_asr = nemo_asr
    return _nemo_asr


# =============================================================================
# Model registry
# =============================================================================

ARCH_REGISTRY = {
    "ctc": {
        "model_cls":   "EncDecCTCModelBPE",
        "pretrained":  "stt_en_conformer_ctc_large",
        "val_monitor": "val_wer",
        "description": "Conformer CTC",
    },
    "rnnt": {
        "model_cls":   "EncDecRNNTBPEModel",
        "pretrained":  "stt_en_conformer_transducer_large",
        "val_monitor": "val_wer",
        "description": "Conformer RNNT",
    },
    "hybrid": {
        "model_cls":   "EncDecHybridRNNTCTCBPEModel",
        "pretrained":  "stt_en_fastconformer_hybrid_large_pc",
        "val_monitor": "val_wer",
        "description": "FastConformer Hybrid RNNT-CTC",
    },
}


# =============================================================================
# Parameter utilities
# =============================================================================

def _count_params(module) -> tuple[int, int]:
    """Return (trainable_params, total_params) for a module."""
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


def print_trainable_summary(model) -> None:
    """
    Print a per-component breakdown showing trainable vs frozen parameters.

    Components shown:
      encoder          Conformer acoustic encoder
      decoder          RNN-T prediction network (language model side)
      joint            RNN-T joint network
      (ctc_decoder)    CTC output projection (hybrid models only)
    """
    components: dict[str, object] = {}
    if hasattr(model, "encoder"):
        components["encoder (Conformer)"] = model.encoder
    if hasattr(model, "decoder"):
        components["decoder (prediction net)"] = model.decoder
    if hasattr(model, "joint"):
        components["joint network"] = model.joint
    if hasattr(model, "ctc_decoder"):
        components["ctc_decoder"] = model.ctc_decoder

    print("\n" + "─" * 64)
    print(f"  {'Component':<32} {'Trainable':>12} {'Total':>12}")
    print("─" * 64)
    for name, mod in components.items():
        tr, tot = _count_params(mod)
        tag = "" if tr > 0 else "  [FROZEN]"
        print(f"  {name:<32} {tr:>12,} {tot:>12,}{tag}")
    tr_all, tot_all = _count_params(model)
    print("─" * 64)
    print(f"  {'TOTAL':<32} {tr_all:>12,} {tot_all:>12,}")
    print("─" * 64 + "\n")


# =============================================================================
# Encoder freeze helpers
# =============================================================================

def _get_encoder_blocks(encoder) -> list | None:
    """
    Return the list of Conformer blocks inside the encoder.
    NeMo uses different attribute names across versions.
    """
    for attr in ("layers", "conformer_layers", "blocks"):
        if hasattr(encoder, attr):
            blocks = getattr(encoder, attr)
            if hasattr(blocks, "__len__") and len(blocks) > 0:
                return list(blocks)
    return None


def apply_freeze_strategy(
    model,
    decoder_only: bool,
    unfreeze_top_n_layers: int,
) -> None:
    """
    Apply the chosen freeze strategy immediately after model load.

    decoder_only=True
        Freeze entire encoder.  Train only prediction network + joint.
        Best for vocabulary / keyword boosting (SC legal jargon, names).
        ~5M trainable params out of ~91M total.

    unfreeze_top_n_layers=K  (with decoder_only=False)
        Freeze entire encoder, then unfreeze the top K Conformer blocks.
        Good middle ground: encoder top layers adapt accent/speaking style
        while keeping lower acoustic layers frozen.

    Neither flag
        Full fine-tuning (encoder warm-up handled by UnfreezeEncoderCallback).
    """
    if decoder_only:
        model.encoder.freeze()
        print("\n[freeze] decoder-only: encoder permanently frozen.")
        print_trainable_summary(model)
        return

    if unfreeze_top_n_layers > 0:
        model.encoder.freeze()
        blocks = _get_encoder_blocks(model.encoder)
        if blocks is None:
            print(
                f"\n[freeze] WARNING: could not locate Conformer blocks; "
                f"entire encoder remains frozen.\n"
                f"         Try --decoder-only instead."
            )
        else:
            n = min(unfreeze_top_n_layers, len(blocks))
            for block in blocks[-n:]:
                for p in block.parameters():
                    p.requires_grad = True
            print(
                f"\n[freeze] top-{n}-layers: unfroze top {n} of "
                f"{len(blocks)} Conformer blocks."
            )
        print_trainable_summary(model)


# =============================================================================
# Callbacks
# =============================================================================

class UnfreezeEncoderCallback(pl.Callback):
    """
    Freeze the encoder at training start; unfreeze at `unfreeze_epoch`.

    Only used when neither --decoder-only nor --unfreeze-top-n-layers is set.
    Lets prediction net + joint adapt first, then unlocks the full network.
    """

    def __init__(self, unfreeze_epoch: int = 1):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self._frozen = False

    def on_train_start(self, trainer, pl_module):
        if self.unfreeze_epoch > 0 and hasattr(pl_module, "encoder"):
            pl_module.encoder.freeze()
            self._frozen = True
            print(
                f"\n[UnfreezeEncoder] Encoder frozen for epoch(s) 0–"
                f"{self.unfreeze_epoch - 1}. Will unfreeze at epoch "
                f"{self.unfreeze_epoch}."
            )
            print_trainable_summary(pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        if self._frozen and trainer.current_epoch >= self.unfreeze_epoch:
            if hasattr(pl_module, "encoder"):
                pl_module.encoder.unfreeze()
            self._frozen = False
            print(
                f"\n[UnfreezeEncoder] Epoch {trainer.current_epoch}: "
                f"encoder unfrozen — full fine-tuning begins."
            )
            print_trainable_summary(pl_module)


# =============================================================================
# Model loading
# =============================================================================

def load_model(arch: str, pretrained_override: str = ""):
    nemo_asr   = get_nemo_asr()
    cfg        = ARCH_REGISTRY[arch]
    model_cls  = getattr(nemo_asr.models, cfg["model_cls"])
    model_name = pretrained_override or cfg["pretrained"]
    print(f"\nLoading pretrained model: {model_name}  ({cfg['description']})")
    return model_cls.from_pretrained(model_name)


# =============================================================================
# Dataset configuration
# =============================================================================

def configure_datasets(
    model,
    train_manifest: str,
    val_manifest: str,
    batch_size: int,
    num_workers: int,
) -> None:
    """
    Wire NeMo's train/val data loaders.

    Key overrides:
      sample_rate            — required field; no default in pretrained cfgs.
      is_tarred              — pretrained cfgs often point at TAR-packed buckets
      tarred_audio_filepaths   from the NGC training run; disable for plain WAVs.

    Must be called AFTER model.set_trainer() so NeMo knows world_size etc.
    """
    common = dict(
        sample_rate            = 16000,
        batch_size             = batch_size,
        num_workers            = num_workers,
        pin_memory             = True,
        is_tarred              = False,
        tarred_audio_filepaths = None,
    )

    with open_dict(model.cfg):
        for k, v in {**common, "manifest_filepath": train_manifest, "shuffle": True}.items():
            model.cfg.train_ds[k] = v
        for k, v in {**common, "manifest_filepath": val_manifest, "shuffle": False}.items():
            model.cfg.validation_ds[k] = v

    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)


# =============================================================================
# Optimizer
# =============================================================================

def configure_optimizer(model, lr: float, warmup_steps: int) -> None:
    with open_dict(model.cfg):
        model.cfg.optim.lr           = lr
        model.cfg.optim.name         = "adamw"
        model.cfg.optim.weight_decay = 1e-3
        if hasattr(model.cfg.optim, "sched"):
            model.cfg.optim.sched.warmup_steps = warmup_steps
    model.setup_optimization(model.cfg.optim)


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
                        default="./data/manifests/train.jsonl",
                        help="Path to NeMo JSONL train manifest "
                             "(default: ./data/manifests/train.jsonl).")
    parser.add_argument("--val-manifest",
                        default="./data/manifests/val.jsonl",
                        help="Path to NeMo JSONL validation manifest "
                             "(default: ./data/manifests/val.jsonl).")
    parser.add_argument("--output-dir",     default="./runs/nemo_rnnt",
                        help="Directory to save checkpoints and final.nemo.")

    # ── Architecture ──────────────────────────────────────────────────────────
    parser.add_argument("--arch",       default="rnnt",
                        choices=["ctc", "rnnt", "hybrid"],
                        help="Model architecture (default: rnnt).")
    parser.add_argument("--pretrained", default="",
                        help="Override default NGC pretrained model name.")

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--batch-size",   type=int,   default=8)
    parser.add_argument("--grad-accum",   type=int,   default=4)
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--max-steps",    type=int,   default=-1,
                        help="Cap training steps (use for smoke tests).")
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int,   default=1000)
    parser.add_argument("--num-workers",  type=int,   default=4)
    parser.add_argument("--devices",      type=int,   default=1)
    parser.add_argument("--precision",    default="16-mixed",
                        choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--resume-from",  default="",
                        help="Path to a .ckpt to resume training.")

    # ── Freeze strategy (mutually exclusive) ──────────────────────────────────
    freeze_group = parser.add_mutually_exclusive_group()
    freeze_group.add_argument(
        "--decoder-only", action="store_true", default=False,
        help=(
            "RECOMMENDED for SC domain adaptation.\n"
            "Permanently freeze the Conformer encoder; train ONLY the\n"
            "prediction network (decoder) + joint network.\n"
            "  • ~5M trainable params (vs ~91M total)\n"
            "  • Targets the LM / vocabulary side of RNN-T\n"
            "  • 10-45%% relative WER gain on domain shift (Arxiv 2104.11127)\n"
            "  • No risk of degrading general English acoustics\n"
        ),
    )
    freeze_group.add_argument(
        "--unfreeze-top-n-layers", type=int, default=0,
        metavar="N",
        help=(
            "Freeze encoder but unfreeze the top N Conformer blocks.\n"
            "Use if decoder-only plateaus and accent adaptation is needed.\n"
            "Typical values: 4 (light), 8 (moderate), 12 (aggressive)."
        ),
    )
    freeze_group.add_argument(
        "--unfreeze-epoch", type=int, default=0,
        metavar="EPOCH",
        help=(
            "Warm-up then full fine-tuning.\n"
            "Freeze encoder for the first EPOCH epochs, then unfreeze all.\n"
            "0 (default) = no warm-up; train everything from step 1."
        ),
    )

    args = parser.parse_args()

    # ── Sanity checks ─────────────────────────────────────────────────────────
    for f in [args.train_manifest, args.val_manifest]:
        if not Path(f).exists():
            raise FileNotFoundError(
                f"Manifest not found: {f}\n"
                "Run 04_make_manifest.py first."
            )

    use_gpu     = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    precision   = args.precision if use_gpu else "32"
    if not use_gpu:
        print("WARNING: No GPU — training will be very slow. Forcing precision=32.")
        args.devices = 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arch_cfg  = ARCH_REGISTRY[args.arch]
    eff_batch = args.batch_size * args.grad_accum

    if args.decoder_only:
        freeze_desc = "decoder-only (encoder permanently frozen)"
    elif args.unfreeze_top_n_layers > 0:
        freeze_desc = f"partial encoder (top {args.unfreeze_top_n_layers} blocks + decoder)"
    elif args.unfreeze_epoch > 0:
        freeze_desc = f"warm-up (encoder frozen epoch 0–{args.unfreeze_epoch-1}, then full)"
    else:
        freeze_desc = "full (all components from step 1)"

    print("=" * 64)
    print("  SC ASR — NeMo Fine-tuning")
    print(f"  Architecture : {args.arch}  ({arch_cfg['description']})")
    print(f"  Freeze mode  : {freeze_desc}")
    print(f"  Device       : {torch.cuda.get_device_name(0) if use_gpu else 'CPU'}")
    print(f"  Precision    : {precision}")
    print(f"  Batch        : {args.batch_size} × grad_accum {args.grad_accum} = {eff_batch}")
    print(f"  LR           : {args.lr}")
    print(f"  Output       : {out_dir}")
    print("=" * 64)

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(args.arch, args.pretrained)

    # ── Apply freeze strategy ─────────────────────────────────────────────────
    # decoder-only and unfreeze-top-n-layers are applied immediately.
    # unfreeze-epoch is handled by UnfreezeEncoderCallback during training.
    apply_freeze_strategy(
        model,
        decoder_only          = args.decoder_only,
        unfreeze_top_n_layers = args.unfreeze_top_n_layers,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks: list[pl.Callback] = [
        EarlyStopping(monitor="val_wer", patience=3, mode="min"),
        ModelCheckpoint(
            dirpath    = str(out_dir / "checkpoints"),
            monitor    = "val_wer",
            mode       = "min",
            save_top_k = 1,
            save_last  = True,
            filename   = "best-{epoch:02d}-{val_wer:.3f}",
        ),
    ]
    # Only attach warm-up callback when neither static freeze strategy is used
    if (
        not args.decoder_only
        and args.unfreeze_top_n_layers == 0
        and args.unfreeze_epoch > 0
    ):
        callbacks.append(
            UnfreezeEncoderCallback(unfreeze_epoch=args.unfreeze_epoch)
        )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        accelerator             = accelerator,
        devices                 = args.devices,
        max_epochs              = args.epochs,
        max_steps               = args.max_steps,
        accumulate_grad_batches = args.grad_accum,
        precision               = precision,
        callbacks               = callbacks,
        default_root_dir        = str(out_dir),
        log_every_n_steps       = 20,
    )

    # set_trainer BEFORE configure_datasets — NeMo reads world_size from trainer
    model.set_trainer(trainer)

    # ── Datasets & optimizer ──────────────────────────────────────────────────
    configure_datasets(
        model,
        train_manifest = args.train_manifest,
        val_manifest   = args.val_manifest,
        batch_size     = args.batch_size,
        num_workers    = args.num_workers,
    )
    configure_optimizer(model, lr=args.lr, warmup_steps=args.warmup_steps)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit(
        model,
        ckpt_path=args.resume_from or None,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    final_model = out_dir / "final.nemo"
    model.save_to(str(final_model))
    print(f"\nSaved: {final_model}")
    print(f"Next:  python 06_evaluate.py --nemo-model {final_model}")


if __name__ == "__main__":
    main()
