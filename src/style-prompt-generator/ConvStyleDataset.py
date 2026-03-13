from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# sources where the first few turns can be text-only (hdf5_idx == -1)
_TEXT_ONLY_SOURCES = {"styletalk"}
 
# styletalk always has exactly this many text-only turns at the front of a chain
_STYLETALK_TEXT_ONLY_TURNS = 3


class AudioHDF5Dataset(Dataset):
    """
    Streams waveforms from HDF5 alongside metadata. Each item is a
    conversation chain of `num_turns` utterances in chronological order,
    resolved by walking the prev_filename linked list backwards from the
    anchor utterance (turn_index == num_turns-1 ) (zero indexed).

    Rows missing audio (hdf5_idx == -1) are excluded entirely, along with
    any chain that can't be fully resolved due to broken links.
    """

    def __init__(
        self,
        h5_path:      str,
        meta_path:    str,
        meta_columns: Optional[List[str]] = None,
        max_len_sec:  Optional[float]     = None,
        sample_rate:  int                 = 16_000,
        num_turns:    int                 = 5,
        transform=None,
    ):
        self.h5_path    = Path(h5_path)
        self.meta_path  = Path(meta_path)
        self.transform  = transform
        self.sr         = sample_rate
        self.max_len    = int(max_len_sec * sample_rate) if max_len_sec else None
        self.num_turns  = num_turns

        meta = pd.read_parquet(meta_path)
        
        # drop hard errors only; text-only rows (-1) are handled per-source below
        meta = meta[meta["hdf5_idx"] >= -1].reset_index(drop=True)

        # index by relative_audio_path so prev_filename lookups are O(1)
        self._path_to_row = meta.set_index("relative_audio_path")

        # anchor rows are the "last" turn in each chain we want to load
        anchors = meta[meta["turn_index"] == (num_turns - 1)].reset_index(drop=True)

        # resolve every anchor into a full chain, drop any with broken links
        if meta_columns is not None:
            # always keep the fields we need for chain walking + audio loading
            required = {"hdf5_idx", "relative_audio_path", "prev_filename", "turn_index"}
            self._extra_cols = [c for c in meta_columns if c not in required]
        else:
            required = {"hdf5_idx", "relative_audio_path", "prev_filename", "turn_index"}
            self._extra_cols = [c for c in meta.columns if c not in required]

        self._chains = self._build_chains(anchors)
        self._h5 = None  # lazy-open per worker so forking doesn't break the fd

    def _build_chains(self, anchors: pd.DataFrame) -> List[List[dict]]:
        chains = []
        for _, anchor_row in anchors.iterrows():
            chain = self._walk_chain(anchor_row)
            if chain is not None:
                chains.append(chain)
        return chains

    def _walk_chain(self, anchor_row) -> Optional[List[dict]]:
        # walk backwards num_turns times, collecting rows oldest-first
        chain = []
        current = anchor_row

        for _ in range(self.num_turns):
            chain.append(current)
            prev_path = current.get("prev_filename")

            # stop early if we're at the start of a conversation
            if pd.isna(prev_path) or prev_path == "":
                break

            if prev_path not in self._path_to_row.index:
                # broken link -- skip this chain entirely
                return None

            current = self._path_to_row.loc[prev_path]

        # reverse so the chain runs chronologically oldest -> newest
        chain.reverse()

        # only keep full chains -- shorter ones are incomplete conversations
        if len(chain) < self.num_turns:
            return None
 
        source = str(anchor_row.get("source", "")).lower()
 
        if source == "styletalk":
            return self._validate_styletalk_chain(chain)
        else:
            return self._validate_expresso_chain(chain)
        

    def _validate_styletalk_chain(self, chain: list) -> Optional[list]:
        # styletalk: first N turns must be text-only, the rest must have audio
        for turn_pos, row in enumerate(chain):
            hdf5_idx = int(row["hdf5_idx"])
            is_text_only_slot = turn_pos < _STYLETALK_TEXT_ONLY_TURNS
 
            if is_text_only_slot and hdf5_idx != -1:
                # we expect text-only here; a real audio file is unexpected but not fatal
                pass
            elif not is_text_only_slot and hdf5_idx == -1:
                # audio turns must actually have audio
                return None
 
        return chain
 
    def _validate_expresso_chain(self, chain: list) -> Optional[list]:
        # expresso: no text-only rows allowed anywhere in the chain
        for row in chain:
            if int(row["hdf5_idx"]) == -1:
                return None
        return chain

    def _get_h5(self):
        # open lazily so each DataLoader worker gets its own file handle
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5

    def _load_waveform(self, hdf5_idx: int):
        hf  = self._get_h5()
        key = f"audio/{hdf5_idx:06d}"
        ds  = hf[key]
        wav = ds[()]
        sr  = int(ds.attrs["sample_rate"])
        return wav, sr

    def _pad_or_trim(self, wav: np.ndarray) -> np.ndarray:
        if self.max_len is None:
            return wav
        if len(wav) >= self.max_len:
            return wav[: self.max_len]
        return np.pad(wav, (0, self.max_len - len(wav)))

    def __len__(self):
        return len(self._chains)

    def __getitem__(self, i):
        chain = self._chains[i]
 
        utterances = []
        for row in chain:
            hdf5_idx  = int(row["hdf5_idx"])
            is_text_only = hdf5_idx == -1
 
            if is_text_only:
                # no waveform to load; downstream should check this flag
                wav_tensor  = None
                sample_rate = self.sr
            else:
                wav, sample_rate = self._load_waveform(hdf5_idx)
 
                if self.transform is not None:
                    wav = self.transform(wav)
 
                wav        = self._pad_or_trim(wav)
                wav_tensor = torch.from_numpy(wav)
 
            utt = {
                "audio":               wav_tensor,
                "sample_rate":         sample_rate,
                "hdf5_idx":            hdf5_idx,
                "turn_index":          int(row["turn_index"]),
                "text_only":           is_text_only,
                "relative_audio_path": row.name,  # relative_audio_path is the index
            }
 
            for col in self._extra_cols:
                val = row.get(col, "")
                if isinstance(val, float) and np.isnan(val):
                    val = ""
                utt[col] = val
 
            utterances.append(utt)
 
        return utterances  # list of dicts, chronological order

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def collate_pad(batch):
    """
    Collates a batch of conversation chains. Each item in `batch` is a list
    of utterance dicts (one per turn). Returns a dict where 'audio' is
    (B, T, max_wav_len), 'lengths' is (B, T), and 'text_only' is (B, T) bool.
    Text-only turns (audio=None) are left as zero-padded rows in the tensor.
 
    Use as: DataLoader(..., collate_fn=collate_pad)
    """
    B = len(batch)
    T = len(batch[0])
 
    # only consider turns that actually have audio when computing max length
    audio_lengths = [
        utt["audio"].shape[0]
        for chain in batch
        for utt in chain
        if utt["audio"] is not None
    ]
    max_len = max(audio_lengths) if audio_lengths else 0
 
    padded    = torch.zeros(B, T, max_len, dtype=torch.float32)
    lengths   = torch.zeros(B, T, dtype=torch.long)
    text_only = torch.zeros(B, T, dtype=torch.bool)
 
    for b, chain in enumerate(batch):
        for t, utt in enumerate(chain):
            if utt["audio"] is not None:
                L = utt["audio"].shape[0]
                padded[b, t, :L] = utt["audio"]
                lengths[b, t]    = L
            text_only[b, t] = utt["text_only"]
 
    meta_keys = [k for k in batch[0][0] if k not in ("audio", "text_only")]
    out = {k: [[utt[k] for utt in chain] for chain in batch] for k in meta_keys}
 
    out["audio"]     = padded     # (B, T, max_wav_len)
    out["lengths"]   = lengths    # (B, T)
    out["text_only"] = text_only  # (B, T)
    return out