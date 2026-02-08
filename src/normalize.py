from pathlib import Path
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import shutil


def normalize_file(input_path: Path, input_root: Path, output_root: Path):
    """
    Normalize a single WAV file using SoX, writing to the output directory
    while preserving relative paths.
    """
    # relative path from input root
    rel_path = input_path.relative_to(input_root)
    output_path = output_root / rel_path

    # ensure output folder exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # backup path in the output folder
    backup_path = output_path.with_suffix(output_path.suffix + ".backup")

    # skip if already normalized
    if output_path.exists():
        return

    # copy input to backup first
    shutil.copy2(input_path, backup_path)

    cmd = [
        "sox",
        str(backup_path),
        str(output_path),
        "norm",
        "-0.1",
    ]

    subprocess.run(cmd, check=True)


def normalize_directory(
    input_root: str,
    output_root: str,
    workers: int | None = None,
):
    """
    Normalize all WAV files from input_root into output_root in parallel.
    """
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()

    wav_files = list(input_root.rglob("*.wav"))

    if workers is None:
        workers = max(os.cpu_count() - 1, 1)

    print(f"Normalizing {len(wav_files)} files using {workers} workers...")
    
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(normalize_file, wav, input_root, output_root)
            for wav in wav_files
        ]

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Normalizing", unit="file"):
            pass


if __name__ == "__main__":
    input_dir = r"data/raw/paraspeechcaps/audio/expresso"
    output_dir = r"data/preprocessed/paraspeechcaps/audio/expresso_normalized"
    normalize_directory(input_dir, output_dir)
