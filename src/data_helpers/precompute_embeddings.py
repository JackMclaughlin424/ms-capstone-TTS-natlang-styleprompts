"""
Precomputes WavLM frame-level hidden states and BERT token hidden states
for every utterance in the dataset and writes them to a new HDF5 cache.

Eliminates the heavy encoder forward passes from the training loop.
SelfAttentivePooling (and optionally wavlm_layer_weights) are still learned
at training time from the cached frame-level features.

Output HDF5 layout (mirrors existing audio/{idx:06d} convention):
  /wavlm/{hdf5_idx:06d}     float16  (T_frames, 768)
      mean over 13 WavLM hidden layers; written for audio utterances only
  /bert/{bert_emb_idx:06d}  float16  (seq_len, 768)
      BERT last_hidden_state; written for every utterance (audio + text-only)

A companion parquet is also written alongside the cache with the extra
`bert_emb_idx` column so ConvoStyleDataset can resolve BERT lookups.

Usage:
  python precompute_embeddings.py \\
      --h5_path /data/merged_audio_full.h5 \\
      --meta_path /data/merged_metadata_full.parquet \\
      --output_h5 /data/merged_embeddings.h5 \\
      --output_meta /data/merged_metadata_full_emb.parquet \\
      [--wavlm_repo microsoft/wavlm-base-plus] \\
      [--bert_repo google-bert/bert-base-uncased] \\
      [--batch_size 32] \\
      [--max_len_sec 15] \\
      [--device cuda]

  Or pass a training config JSON and let the script read the paths:
  python precompute_embeddings.py --config default_config.json
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertModel,
    Wav2Vec2FeatureExtractor,
    WavLMModel,
)



# helpers

def _mean_wavlm_hidden(hidden_states: tuple) -> torch.Tensor:
    """Stack all WavLM hidden states and return their mean over the layer axis.

    hidden_states: tuple of L tensors each (B, T, 768)
    returns: (B, T, 768) float32
    """
    stacked = torch.stack(hidden_states, dim=0)   # (L, B, T, 768)
    return stacked.mean(dim=0)                     # (B, T, 768)


def _write_float16(grp: h5py.Group, key: str, arr: np.ndarray) -> None:
    """Write array as float16; skip if key already exists (resume support)."""
    if key in grp:
        return
    grp.create_dataset(key, data=arr.astype(np.float16), compression="lzf")



# WavLM embedding

@torch.no_grad()
def compute_wavlm_embeddings(
    h5_src: h5py.File,
    h5_dst: h5py.Group,
    audio_rows: pd.DataFrame,
    processor: Wav2Vec2FeatureExtractor,
    model: WavLMModel,
    max_len_samples: int,
    batch_size: int,
    device: torch.device,
) -> None:
    """Iterate over audio rows in batches, run WavLM, write to h5_dst."""
    rows = list(audio_rows.itertuples())
    total = len(rows)

    for start in tqdm(range(0, total, batch_size), desc="WavLM", unit="batch"):
        batch_rows = rows[start : start + batch_size]

        # skip batch if all keys already written
        if all(f"{int(r.hdf5_idx):06d}" in h5_dst for r in batch_rows):
            continue

        # load waveforms
        wavs = []
        for r in batch_rows:
            key = f"audio/{int(r.hdf5_idx):06d}"
            wav = h5_src[key][()].astype(np.float32)
            # trim / pad to uniform length
            if len(wav) >= max_len_samples:
                wav = wav[:max_len_samples]
            else:
                wav = np.pad(wav, (0, max_len_samples - len(wav)))
            wavs.append(wav)

        inputs = processor(
            wavs,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values  = inputs.input_values.to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        out = model(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # mean over L layers → (B, T_frames, 768)
        hidden = _mean_wavlm_hidden(out.hidden_states)   # float32

        # trim frame dimension to what the attention mask covers (removes padding frames)
        if attention_mask is not None:
            # WavLM downsamples by ~320; compute valid frame count per item
            feat_extract_len = model._get_feat_extract_output_lengths(
                attention_mask.sum(dim=1)
            )  # (B,)
        else:
            feat_extract_len = [hidden.shape[1]] * len(batch_rows)

        for i, r in enumerate(batch_rows):
            key = f"{int(r.hdf5_idx):06d}"
            t   = int(feat_extract_len[i])
            arr = hidden[i, :t, :].cpu().numpy()         # (T_valid, 768)
            _write_float16(h5_dst, key, arr)



# BERT embedding

@torch.no_grad()
def compute_bert_embeddings(
    h5_dst: h5py.Group,
    rows: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: BertModel,
    batch_size: int,
    device: torch.device,
    max_length: int = 128,
) -> None:
    """Tokenize transcriptions and run BERT; write last_hidden_state to h5_dst."""
    row_list = list(rows.itertuples())
    total    = len(row_list)

    for start in tqdm(range(0, total, batch_size), desc="BERT", unit="batch"):
        batch_rows = row_list[start : start + batch_size]

        # skip batch if all keys already written
        if all(f"{int(r.bert_emb_idx):06d}" in h5_dst for r in batch_rows):
            continue

        texts = [str(getattr(r, "transcription", "") or "") for r in batch_rows]

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        last_hidden = out.last_hidden_state   # (B, seq_len, 768)

        for i, r in enumerate(batch_rows):
            key      = f"{int(r.bert_emb_idx):06d}"
            seq_len  = int(enc["attention_mask"][i].sum())    # exclude padding
            arr      = last_hidden[i, :seq_len, :].cpu().numpy()  # (seq_len, 768)
            _write_float16(h5_dst, key, arr)



# main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute WavLM + BERT embeddings")
    p.add_argument("--config",       type=str, default=None,
                   help="Path to training JSON config (overrides individual path flags)")
    p.add_argument("--h5_path",      type=str, default=None)
    p.add_argument("--meta_path",    type=str, default=None)
    p.add_argument("--output_h5",    type=str, default=None,
                   help="Destination HDF5 (default: <h5_path stem>_embeddings.h5)")
    p.add_argument("--output_meta",  type=str, default=None,
                   help="Destination parquet with bert_emb_idx column "
                        "(default: <meta_path stem>_emb.parquet)")
    p.add_argument("--wavlm_repo",   type=str, default="microsoft/wavlm-base-plus")
    p.add_argument("--bert_repo",    type=str, default="google-bert/bert-base-uncased")
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--max_len_sec",  type=float, default=15.0)
    p.add_argument("--sample_rate",  type=int,   default=16_000)
    p.add_argument("--device",       type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- resolve paths (config overrides CLI flags) ----
    if args.config is not None:
        with open(args.config) as f:
            cfg = json.load(f)
        h5_path   = cfg.get("h5_path",   args.h5_path)
        meta_path = cfg.get("meta_path", args.meta_path)
        args.max_len_sec  = cfg.get("max_len_sec",  args.max_len_sec)
        args.sample_rate  = cfg.get("sample_rate",  args.sample_rate)
    else:
        h5_path   = args.h5_path
        meta_path = args.meta_path

    if not h5_path or not meta_path:
        sys.exit("ERROR: provide --h5_path and --meta_path (or --config)")

    h5_path   = Path(h5_path)
    meta_path = Path(meta_path)

    output_h5   = Path(args.output_h5)   if args.output_h5   else h5_path.with_name(h5_path.stem   + "_embeddings.h5")
    output_meta = Path(args.output_meta) if args.output_meta else meta_path.with_name(meta_path.stem + "_emb.parquet")

    device          = torch.device(args.device)
    max_len_samples = int(args.max_len_sec * args.sample_rate)

    print(f"Source audio HDF5 : {h5_path}")
    print(f"Source metadata   : {meta_path}")
    print(f"Output HDF5       : {output_h5}")
    print(f"Output metadata   : {output_meta}")
    print(f"Device            : {device}")

    # ---- load metadata ----
    meta = pd.read_parquet(meta_path)
    meta = meta[meta["hdf5_idx"] >= -1].reset_index(drop=True)

    # add sequential bert_emb_idx if not already present
    if "bert_emb_idx" not in meta.columns:
        meta["bert_emb_idx"] = np.arange(len(meta), dtype=np.int64)

    # ---- load models ----
    print(f"\nLoading WavLM from {args.wavlm_repo} ...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.wavlm_repo)
    wavlm     = WavLMModel.from_pretrained(args.wavlm_repo).to(device).eval()

    print(f"Loading BERT from {args.bert_repo} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_repo)
    bert      = BertModel.from_pretrained(args.bert_repo).to(device).eval()

    # ---- open HDF5 files ----
    audio_rows = meta[meta["hdf5_idx"] >= 0]
    print(f"\nAudio utterances  : {len(audio_rows):,}")
    print(f"Total utterances  : {len(meta):,}  (includes text-only)")

    with h5py.File(h5_path,   "r", swmr=True) as h5_src, \
         h5py.File(output_h5, "a")            as h5_dst:

        wavlm_grp = h5_dst.require_group("wavlm")
        bert_grp  = h5_dst.require_group("bert")

        # ---- WavLM ----
        print("\n=== WavLM ===")
        compute_wavlm_embeddings(
            h5_src        = h5_src,
            h5_dst        = wavlm_grp,
            audio_rows    = audio_rows,
            processor     = processor,
            model         = wavlm,
            max_len_samples = max_len_samples,
            batch_size    = args.batch_size,
            device        = device,
        )

        # ---- BERT ----
        print("\n=== BERT ===")
        compute_bert_embeddings(
            h5_dst     = bert_grp,
            rows       = meta,
            tokenizer  = tokenizer,
            model      = bert,
            batch_size = args.batch_size,
            device     = device,
        )

    # ---- save updated metadata ----
    meta.to_parquet(output_meta, index=True)
    print(f"\nSaved updated metadata → {output_meta}")
    print("Done.")


if __name__ == "__main__":
    main()
