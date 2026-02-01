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


def clone_styletalk(path):
    """
    Downloads StyleTalk files from Google Drive
    """
    data_dir = Path(path)
    annot_dir = data_dir / "annotations"
    # Cloning annotations
    if annot_dir.exists():
        print(f"StyleTalk annotations already exists at {annot_dir}")
        
    else:
        # LibriTTS-P (extra annotation data)
        subprocess.run(
            [
                "git", "clone",
                "https://github.com/DanielLin94144/StyleTalk.git",
                str(annot_dir),
            ],
            check=True,
        )
        
    # Downloading audio
    

    url = "https://drive.google.com/uc?id=12mlGaZkkBebk4gf0qY684W_bwvmCsu_t"
    name = "audio"
    archive_path = data_dir / f"{name}.tar.gz"
    
    extract_dir = data_dir
    tar_top_folder = "audio" # this is within the tar, so i pass just extract_dir to get correct structure
    final_structure = extract_dir / tar_top_folder 
    
    if final_structure.exists():
        print(f"StyleTalk audio already exists at {final_structure}")
    elif  archive_path.exists():
        print(f"StyleTalk audio tar already exists at {archive_path}")
    else:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        
        import gdown
        gdown.download(url,  str(archive_path), quiet=False)

    # extract
    
    if final_structure.exists():
        print(f"StyleTalk audio already extracted at {final_structure}")
    else:
        extract_tar_gz(archive_path, extract_dir)



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
        "--styletalk",
        action="store_true",
        help="Download the placeholder dataset",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download the placeholder dataset",
    )

    args = parser.parse_args()

    if not (args.libritts or args.styletalk or args.all):
        parser.error("Please specify at least one dataset to download.")

    if args.libritts:
        clone_libritts_p("data/raw/libritts")

    elif args.styletalk:
        clone_styletalk("data/raw/styletalk")
        
    elif args.all:
        clone_libritts_p("data/raw/libritts")
        clone_styletalk("data/raw/styletalk")


if __name__ == "__main__":
    main()
