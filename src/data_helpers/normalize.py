from pathlib import Path
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import shutil


def normalize_file(input_path: Path, input_root: Path, output_root: Path):
    rel_path = input_path.relative_to(input_root)
    output_path = output_root / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return

    # run directly from source no need for a backup since we're not overwriting
    cmd = ["sox", str(input_path), str(output_path), "norm", "-0.1"]
    subprocess.run(cmd, check=True)


def normalize_directory(input_root: str, output_root: str, workers: int | None = 6):
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()

    # follow symlinks to match bash's `find -L` behavior
    wav_files = [
        Path(dirpath) / fname
        for dirpath, dirnames, filenames in os.walk(input_root, followlinks=True)
        for fname in filenames
        if fname.endswith(".wav")
    ]

    if workers is None:
        workers = max(os.cpu_count() - 1, 1)

    print(f"Normalizing {len(wav_files)} files using {workers} workers...")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(normalize_file, wav, input_root, output_root) for wav in wav_files]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Normalizing", unit="file"):
            pass


if __name__ == "__main__":
    input_dir = r"../data/raw/paraspeechcaps/audio/expresso/audio_48khz/conversational_vad_segmented"
    output_dir = r"../data/processed/paraspeechcaps/expresso/audio_48khz/conversational_vad_segmented"
    normalize_directory(input_dir, output_dir)
