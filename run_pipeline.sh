#!/usr/bin/env bash
# =============================================================================
#  Supreme Court ASR – end-to-end pipeline
# =============================================================================
#
#  Prerequisites
#  -------------
#  1. Python 3.10+
#  2. CUDA-capable GPU (≥ 24 GB VRAM for NeMo Hybrid / Conformer-Transducer)
#  3. ffmpeg   → brew install ffmpeg  (macOS)  |  apt install ffmpeg  (Linux)
#  4. NeMo + PyTorch (must be installed before requirements.txt):
#       pip install torch==2.2.* --extra-index-url https://download.pytorch.org/whl/cu121
#       pip install nemo_toolkit[asr]==1.23
#  5. pip install -r requirements.txt
#
#  Quick smoke test (2 hearings, 5 training steps):
#     LIMIT=2 MAX_STEPS=5 bash run_pipeline.sh
#
#  Full run:
#     bash run_pipeline.sh
#
#  Environment variables you can override:
#     DATA_DIR        ./data          root for raw data, segments, manifests
#     RUNS_DIR        ./runs          training output directory
#     LIMIT           0               max hearings to download (0 = all)
#     HEARING         ""              single hearing slug (overrides LIMIT)
#     MAX_STEPS       -1              training steps override (-1 = use epochs)
#     DEVICE          cuda            cuda or cpu
#     WHISPER_MODEL   large-v3        WhisperX model size (for Step 3 alignment)
#     NEMO_ARCH       hybrid          NeMo architecture: ctc | rnnt | hybrid
#     NEMO_PRETRAINED ""              Override NeMo pretrained model name
#     NEMO_BATCH      16              Batch size for NeMo training
#     NEMO_BASELINES  stt_en_conformer_ctc_large,stt_en_conformer_transducer_large
#                                     Zero-shot NeMo baselines for evaluation
#     WHISPER_BASELINE openai/whisper-large-v3
#                                     Zero-shot Whisper baseline for evaluation
# =============================================================================
set -euo pipefail

PY="${PY:-python3}"
DATA_DIR="${DATA_DIR:-./data}"
RUNS_DIR="${RUNS_DIR:-./runs}"
LIMIT="${LIMIT:-0}"
HEARING="${HEARING:-}"
MAX_STEPS="${MAX_STEPS:--1}"
DEVICE="${DEVICE:-cuda}"
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"

# NeMo training config (Step 5)
NEMO_ARCH="${NEMO_ARCH:-hybrid}"
NEMO_PRETRAINED="${NEMO_PRETRAINED:-}"
NEMO_BATCH="${NEMO_BATCH:-16}"

# Evaluation baselines (Step 6)
NEMO_BASELINES="${NEMO_BASELINES:-stt_en_conformer_ctc_large,stt_en_conformer_transducer_large}"
WHISPER_BASELINE="${WHISPER_BASELINE:-openai/whisper-large-v3}"

echo "========================================================"
echo "  Supreme Court ASR Pipeline"
echo "  Data dir  : $DATA_DIR"
echo "  Runs dir  : $RUNS_DIR"
echo "  Device    : $DEVICE"
echo "  WhisperX  : $WHISPER_MODEL  (alignment only)"
echo "  NeMo arch : $NEMO_ARCH"
echo "========================================================"

# ─── Step 1: Download ────────────────────────────────────────────────────────
echo ""
echo "[1/6] Downloading audio (Dropbox MP3) and transcripts (PDF) …"

LIMIT_ARG=""
[[ "${LIMIT}" -gt 0 ]] && LIMIT_ARG="--limit ${LIMIT}"

"$PY" 01_download.py \
    --out-dir "${DATA_DIR}/raw" \
    --include-continuations \
    ${LIMIT_ARG}

# ─── Step 2: Parse transcript PDFs ───────────────────────────────────────────
echo ""
echo "[2/6] Parsing TERES transcript PDFs → clean utterance files …"

HEARING_ARG=""
[[ -n "${HEARING}" ]] && HEARING_ARG="--hearing ${HEARING}"

"$PY" 02_parse_transcript.py \
    --raw-dir "${DATA_DIR}/raw" \
    ${HEARING_ARG}

# ─── Step 3: Align and segment ───────────────────────────────────────────────
echo ""
echo "[3/6] Aligning audio with transcripts (WhisperX forced alignment) …"

"$PY" 03_align_segment.py \
    --raw-dir       "${DATA_DIR}/raw" \
    --out-dir       "${DATA_DIR}/segments" \
    --device        "${DEVICE}" \
    --whisper-model "${WHISPER_MODEL}" \
    ${HEARING_ARG}

# ─── Step 4: Build NeMo manifests ────────────────────────────────────────────
echo ""
echo "[4/6] Filtering and splitting into train / val / test manifests …"

"$PY" 04_make_manifest.py \
    --segments-json "${DATA_DIR}/segments/all_segments.json" \
    --out-dir       "${DATA_DIR}/manifests"

# ─── Step 5: Fine-tune with NeMo ─────────────────────────────────────────────
echo ""
echo "[5/6] Fine-tuning NeMo Conformer (arch: ${NEMO_ARCH}) …"

MAX_STEPS_ARG=""
[[ "${MAX_STEPS}" -gt 0 ]] && MAX_STEPS_ARG="--max-steps ${MAX_STEPS}"

PRETRAINED_ARG=""
[[ -n "${NEMO_PRETRAINED}" ]] && PRETRAINED_ARG="--pretrained ${NEMO_PRETRAINED}"

"$PY" 05_nemo_train.py \
    --train-manifest "${DATA_DIR}/manifests/train.jsonl" \
    --val-manifest   "${DATA_DIR}/manifests/val.jsonl" \
    --output-dir     "${RUNS_DIR}/nemo_${NEMO_ARCH}" \
    --arch           "${NEMO_ARCH}" \
    --batch-size     "${NEMO_BATCH}" \
    --grad-accum     4 \
    --epochs         5 \
    --lr             1e-4 \
    --warmup-steps   1000 \
    --unfreeze-epoch 1 \
    ${PRETRAINED_ARG} \
    ${MAX_STEPS_ARG}

# ─── Step 6: Evaluate ────────────────────────────────────────────────────────
echo ""
echo "[6/6] Evaluating (fine-tuned + baselines) — WER / CER / MER / WIL …"

"$PY" 06_evaluate.py \
    --manifest          "${DATA_DIR}/manifests/test.jsonl" \
    --nemo-model        "${RUNS_DIR}/nemo_${NEMO_ARCH}/final.nemo" \
    --nemo-baselines    "${NEMO_BASELINES}" \
    --whisper-baselines "${WHISPER_BASELINE}" \
    --device            "${DEVICE}" \
    --output-json       "${RUNS_DIR}/eval_results.json"

echo ""
echo "========================================================"
echo "  Pipeline complete!"
echo "  Fine-tuned model : ${RUNS_DIR}/nemo_${NEMO_ARCH}/final.nemo"
echo "  Eval results     : ${RUNS_DIR}/eval_results.json"
echo "========================================================"
