import argparse
import subprocess
from pathlib import Path
import requests
import tarfile
from tqdm import tqdm

def extract_tar_gz(archive_path: Path, extract_to: Path):
    if extract_to.exists():
        print(f"Archive already extracted to {extract_to}")
        return

    print(f"Extracting {archive_path}")
    extract_to.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to, filter="data")
        

def download_file(url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"File already exists: {output_path}")
        return

    print(f"Downloading {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("Content-Length", 0))
        chunk_size = 8192

        with open(output_path, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {output_path.name}",
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    pbar.update(len(chunk))
                

def download_tar_dataset(url: str, download_path, name: str):
    raw_dir = Path(download_path)
    archive_path = raw_dir / f"{name}.tar.gz"
    extract_dir = raw_dir / name

    download_file(url, archive_path)
    extract_tar_gz(archive_path, extract_dir)


def clone_libritts_p(path):
    data_dir = Path(path+"/libritts-p")

    if data_dir.exists():
        print(f"LibriTTS-P already exists at {data_dir}")
    else:
        # LibriTTS-P (extra annotation data)
        subprocess.run(
            [
                "git", "clone",
                "https://github.com/line/LibriTTS-P.git",
                str(data_dir),
            ],
            check=True,
        )
    
    # LibriTTS-R (actual audio files)
    data_dir_R = Path(path+"/libritts-r")
    
    url = "https://openslr.trmal.net/resources/141/dev_clean.tar.gz"
    name = "dev_clean"
    download_tar_dataset(url, data_dir_R, name)
    
    url = "https://openslr.trmal.net/resources/141/dev_other.tar.gz"
    name = "dev_other"
    download_tar_dataset(url, data_dir_R, name)
    
    ## TODO: add training set 


def clone_placeholder_dataset(path):
    """
    Placeholder for another dataset.
    Replace the repo URL and any post-processing as needed.
    """
    data_dir = Path(path)

    if data_dir.exists():
        print(f"[INFO] Placeholder dataset already exists at {data_dir}")
        return

    # TODO: replace with real dataset repo
    subprocess.run(
        [
            "git", "clone",
            "https://github.com/example/example-dataset.git",
            str(data_dir),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets required for this project."
    )

    parser.add_argument(
        "--libritts",
        action="store_true",
        help="Download the LibriTTS-P & R datasets",
    )

    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Download the placeholder dataset",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download the placeholder dataset",
    )

    args = parser.parse_args()

    if not (args.libritts or args.placeholder or args.all):
        parser.error("Please specify at least one dataset to download.")

    if args.libritts:
        clone_libritts_p("data/raw/libritts")

    elif args.placeholder:
        clone_placeholder_dataset("data/raw/placeholder-dataset")
        
    elif args.all:
        clone_libritts_p("data/raw/libritts")
        clone_placeholder_dataset("data/raw/placeholder-dataset")


if __name__ == "__main__":
    main()
