import subprocess
from pathlib import Path

def clone_libritts_p(path):
    data_dir = Path(path)

    if not data_dir.exists():
        subprocess.run([
            "git", "clone",
            "https://github.com/line/LibriTTS-P.git",
            str(data_dir)
        ], check=True)


clone_libritts_p("../data/raw/libritts-p")