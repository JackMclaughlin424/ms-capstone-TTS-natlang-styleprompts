"""
build_hdf5.py

Usage - Default data should be in place if you've run the prerequisite files
-----
  python build_hdf5.py 
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


# Columns with long free-text – kept in Parquet, not HDF5 attrs
TEXT_COLUMNS = {
    "text_description", "transcription",
    "intrinsic_tags", "situational_tags",
    "basic_tags", "all_tags",
}

# Columns to store as lightweight HDF5 dataset-level attrs (short scalars)
ATTR_COLUMNS = {
    "source", "speakerid", "name", "gender",
    "accent", "pitch", "speaking_rate", "noise",
    "utterance_pitch_mean", "snr", "tag_of_interest",
    "conv_id", "turn_index", "prev_filename", "record_type",
}

# hdf5_idx sentinel values written to metadata.parquet
IDX_TEXT_ONLY = -1   # record_type == 'text_only': no audio, intentional
IDX_ERROR     = -2   # audio expected but missing or unreadable

# for debugging!
DEBUG_MAX_ROW = 100


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Return (waveform float32, sample_rate). Mixes down to mono."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)          # (samples,)
    return mono, sr


def safe_attr(value):
    """Convert a value to something HDF5 attrs will accept."""
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def build(df_path: str, audio_root_PSC: str, audio_root_ST: str,
          out_h5: str, out_meta: str):

    df_path        = Path(df_path)
    audio_root_PSC = Path(audio_root_PSC)
    audio_root_ST  = Path(audio_root_ST)
    out_h5         = Path(out_h5)
    out_meta       = Path(out_meta)

    SOURCE_ROOTS = {
        "expresso": audio_root_PSC,
        "styletalk":  audio_root_ST,
    }

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    # Load DataFrame
    print(f"Loading DataFrame from {df_path} ...")
    df = pd.read_parquet(df_path) if df_path.suffix == ".parquet" else pd.read_csv(df_path)
    n_text_only_expected = (df["record_type"] == "text_only").sum()
    print(f"  {len(df):,} rows total  |  "
          f"text_only: {n_text_only_expected:,}  |  "
          f"audio rows: {len(df) - n_text_only_expected:,}")

    attr_cols_present = ATTR_COLUMNS & set(df.columns)


    # Build HDF5

    hdf5_indices = []
    error_rows   = []
    n_text_only  = 0
    idx          = 0

    
    DEBUG_CUR_ROW = 0
    with h5py.File(out_h5, "w") as hf:
        audio_grp = hf.create_group("audio")
        hf.attrs["source_df"]      = str(df_path)
        hf.attrs["audio_root_PSC"] = str(audio_root_PSC)
        hf.attrs["audio_root_ST"]  = str(audio_root_ST)

        for row_num, row in tqdm(df.iterrows(), total=DEBUG_MAX_ROW if DEBUG_MAX_ROW != -1 else len(df), desc="Writing audio"):
            if DEBUG_CUR_ROW == DEBUG_MAX_ROW: break
            DEBUG_CUR_ROW += 1

            record_type = str(row.get("record_type", "")).strip().lower()
            
            # Text-only row: no audio expected, skip silently 
            if record_type == "text_only":
                hdf5_indices.append(IDX_TEXT_ONLY)
                n_text_only += 1
                continue

            source     = row.get("source", "")
            audio_root = SOURCE_ROOTS.get(source)

            if audio_root is None:
                print(f"  [ERROR] Unknown source '{source}' (row {row_num})")
                hdf5_indices.append(IDX_ERROR)
                error_rows.append(row_num)
                continue

            rel_path  = row.get("relative_audio_path", "")
            full_path = audio_root / rel_path

            if not full_path.exists():
                print(f"  [ERROR] Missing file (row {row_num}): {full_path}")
                hdf5_indices.append(IDX_ERROR)
                error_rows.append(row_num)
                continue

            try:
                waveform, sr = load_wav(full_path)
            except Exception as exc:
                print(f"  [ERROR] Unreadable file (row {row_num}): {full_path} -- {exc}")
                hdf5_indices.append(IDX_ERROR)
                error_rows.append(row_num)
                continue

            key = f"{idx:06d}"
            ds  = audio_grp.create_dataset(
                key,
                data=waveform,
                dtype="float32",
                compression="gzip",
                compression_opts=4,
                chunks=True,
            )
            ds.attrs["sample_rate"]   = sr
            ds.attrs["duration_sec"]  = len(waveform) / sr
            ds.attrs["original_path"] = str(rel_path)
            ds.attrs["df_row"]        = int(row_num)

            for col in attr_cols_present:
                ds.attrs[col] = safe_attr(row[col])

            hdf5_indices.append(idx)
            idx += 1


        hf.attrs["total_audio_samples"] = idx
        hf.attrs["total_text_only"]     = n_text_only


    print(f"\nHDF5 written -> {out_h5}")
    print(f"  Audio samples stored : {idx:,}")
    print(f"  Text-only rows       : {n_text_only:,}  (hdf5_idx = {IDX_TEXT_ONLY})")
    print(f"  Unexpected errors    : {len(error_rows):,}  (hdf5_idx = {IDX_ERROR})")
    if error_rows:
        print(f"  Error row indices    : {error_rows}")

    # Save metadata + index

    if DEBUG_MAX_ROW != -1:
        df = df[:DEBUG_MAX_ROW].copy()
    else:
        df = df.copy()

    df["hdf5_idx"] = hdf5_indices
    df.to_parquet(out_meta, index=False)

    print(f"\nMetadata written -> {out_meta}")
    print(f"  Join key : 'hdf5_idx'")
    print(f"    >= 0  -> audio present in HDF5")
    print(f"    {IDX_TEXT_ONLY}   -> text_only row (use transcription only)")
    print(f"    {IDX_ERROR}   -> audio expected but failed (investigate)")


def parse_args():
    p = argparse.ArgumentParser(description="Build HDF5 audio archive + Parquet metadata index")
    p.add_argument("--df",         default="../data/processed/merged_PSC_StyleTalk_CLEANED.parquet",  help="Path to input DataFrame (.csv or .parquet)")
    p.add_argument("--audio_root_PSC", default="../data/raw/paraspeechcaps/audio/expresso",  help="Root directory to paraspeechcaps that relative_audio_path is relative to")
    p.add_argument("--audio_root_ST", default="../data/raw/styletalk/audio",  help="Root directory to StyleTalk that relative_audio_path is relative to")
    p.add_argument("--out_h5",     default="../data/processed/merged_audio.h5",          help="Output HDF5 path")
    p.add_argument("--out_meta",   default="../data/processed/merged_metadata.parquet",   help="Output Parquet path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.df, args.audio_root_PSC, args.audio_root_ST, args.out_h5, args.out_meta)