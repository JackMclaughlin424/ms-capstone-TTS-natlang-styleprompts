import wave
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from joblib import Parallel, delayed


def get_wav_duration(filepath: Path) -> float:
    try:
        with wave.open(str(filepath), "rb") as f:
            return f.getnframes() / f.getframerate()
    except:
        return pd.NA


def get_wav_pitch_mean(y, sr) -> float:
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500)
        if not voiced_flag.any():
            return np.nan
        return float(np.nanmean(f0[voiced_flag]))
    except:
        return np.nan


def get_wav_snr(y) -> float:
    try:
        rms = librosa.feature.rms(y=y)[0]
        threshold = np.percentile(rms, 30)
        noise_power  = np.mean(rms[rms <= threshold] ** 2)
        signal_power = np.mean(rms[rms >  threshold] ** 2)
        if noise_power <= 0:
            return np.nan
        return float(10 * np.log10(signal_power / noise_power))
    except:
        return np.nan


def extract_features(filepath: Path) -> dict:
    """Load wav once and compute all features."""
    try:
        y, sr = librosa.load(filepath, sr=None)
        return {
            'duration':             get_wav_duration(filepath),
            'utterance_pitch_mean': get_wav_pitch_mean(y, sr),
            'snr':                  get_wav_snr(y),
        }
    except Exception as e:
        print(f"Failed {filepath}: {e}")
        return {
            'duration':             pd.NA,
            'utterance_pitch_mean': np.nan,
            'snr':                  np.nan,
        }


def load_styletalk(base_dir: str = "data/raw/styletalk") -> pd.DataFrame:
    splits = ["train", "eval"]
    dfs = []
    for split in splits:
        path = f"{base_dir}/annotations/{split}.csv"
        dfs.append(pd.read_csv(path, sep=",", encoding="utf-8", na_values=["NA", ""]))
    return pd.concat(dfs, ignore_index=True)


def enrich_with_audio_features(
    df: pd.DataFrame,
    audio_root: str = "data/raw/styletalk",
    audio_col: str = "relative_audio_path",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Add duration, pitch, and SNR columns by processing wav files in parallel."""
    paths = [Path(audio_root) / p for p in df[audio_col]]

    print(f"Extracting features from {len(paths)} files using {n_jobs} workers...")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(extract_features)(p) for p in paths
    )

    features_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, features_df], axis=1)


if __name__ == "__main__":
    st_df = load_styletalk()
    st_df = enrich_with_audio_features(st_df)
    print(st_df[['duration', 'utterance_pitch_mean', 'snr']].describe())