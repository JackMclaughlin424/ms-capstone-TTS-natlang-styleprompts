from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AudioHDF5Dataset(Dataset):
    """
    Streams waveforms from HDF5 and returns them alongside any metadata columns
    you request. Rows with hdf5_idx == -1 (missing audio) are automatically dropped.

    Parameters
    ----------
    h5_path       : path to audio.h5
    meta_path     : path to metadata.parquet
    meta_columns  : list of metadata columns to include in each item.
                    Pass None to include all columns.
    max_len_sec   : if set, waveforms are truncated / zero-padded to this length
                    (assumes all files share the same sample rate)
    sample_rate   : expected sample rate – used for max_len_sec calculation
    transform     : optional callable applied to the raw waveform numpy array
                    before it is converted to a tensor
    """

    def __init__(
        self,
        h5_path:      str,
        meta_path:    str,
        meta_columns: Optional[List[str]] = None,
        max_len_sec:  Optional[float]     = None,
        sample_rate:  int                 = 16_000,
        transform=None,
    ):
        self.h5_path    = Path(h5_path)
        self.meta_path  = Path(meta_path)
        self.transform  = transform
        self.sr         = sample_rate
        self.max_len    = int(max_len_sec * sample_rate) if max_len_sec else None

        # ── Load metadata, drop rows without audio ──
        meta = pd.read_parquet(meta_path)
        meta = meta[meta["hdf5_idx"] >= 0].reset_index(drop=True)

        if meta_columns is not None:
            keep = list(set(meta_columns) & set(meta.columns)) + ["hdf5_idx"]
            meta = meta[keep]

        self.meta = meta
        self.meta_columns = [c for c in meta.columns if c != "hdf5_idx"]

        # Keep HDF5 file handle open for the lifetime of the dataset
        # (safe for multi-worker DataLoader when swmr=True or num_workers=0)
        self._h5 = None   # lazy-open inside __getitem__ (fork-safe)

    # Internal

    def _get_h5(self):
        """Lazy-open per worker (fork-safe)."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5

    def _pad_or_trim(self, wav: np.ndarray) -> np.ndarray:
        if self.max_len is None:
            return wav
        if len(wav) >= self.max_len:
            return wav[: self.max_len]
        return np.pad(wav, (0, self.max_len - len(wav)))

    # Public API

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row      = self.meta.iloc[i]
        hdf5_idx = int(row["hdf5_idx"])
        key      = f"audio/{hdf5_idx:06d}"

        hf  = self._get_h5()
        ds  = hf[key]
        wav = ds[()]                          # numpy float32 array
        sr  = int(ds.attrs["sample_rate"])

        if self.transform is not None:
            wav = self.transform(wav)

        wav = self._pad_or_trim(wav)
        wav_tensor = torch.from_numpy(wav)    # (T,)

        item = {
            "audio":       wav_tensor,
            "sample_rate": sr,
            "hdf5_idx":    hdf5_idx,
        }

        for col in self.meta_columns:
            val = row[col]
            # Convert NaN → empty string for text columns
            if isinstance(val, float) and np.isnan(val):
                val = ""
            item[col] = val

        return item

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


# Collate helper (handles variable-length waveforms without max_len_sec)

def collate_pad(batch):
    """
    Variable-length collate: pads waveforms in a batch to the longest sample.
    Use as: DataLoader(..., collate_fn=collate_pad)
    """
    max_len = max(item["audio"].shape[0] for item in batch)
    B = len(batch)

    padded = torch.zeros(B, max_len, dtype=torch.float32)
    lengths = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        L = item["audio"].shape[0]
        padded[i, :L] = item["audio"]
        lengths[i]    = L

    out = {k: [item[k] for item in batch] for k in batch[0] if k != "audio"}
    out["audio"]   = padded
    out["lengths"] = lengths
    return out