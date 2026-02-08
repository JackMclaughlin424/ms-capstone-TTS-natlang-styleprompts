import argparse
import subprocess
from pathlib import Path
import requests
import tarfile
from tqdm import tqdm

def extract_tar(archive_path: Path, extract_to: Path, gz=True):

    print(f"Extracting {archive_path}")
    extract_to.mkdir(parents=True, exist_ok=True)

    if gz:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_to, filter="data")
    else:
        with tarfile.open(archive_path, "r") as tar:
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
                

def download_tar_dataset(url: str, download_path, name: str, gz=True):
    raw_dir = Path(download_path)
    
    if (raw_dir / name).exists():
        print(f"{name} dataset already downloaded and extracted at {raw_dir / name}")
        return
    
    archive_path = raw_dir / (f"{name}.tar.gz" if gz else f"{name}.tar")
    extract_dir = raw_dir

    download_file(url, archive_path)
    extract_tar(archive_path, extract_dir, gz)


def clone_libritts_p(path):
    """Clone LibriTTS-P and download LibriTTS-R audio
    
    https://github.com/line/LibriTTS-P
    """
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
    Downloads StyleTalk files from Google Drive, clones repo for annotations
    
    https://github.com/DanielLin94144/StyleTalk
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
        extract_tar(archive_path, extract_dir)
        
     
    
        
def clone_paraspeechcaps(path):
    """Download helper scripts, annotations files, and select audio files for ParaSpeechCaps dataset.
    
    **Note**: Does not download all audio for ParaSpeechCaps, only the audio used for this project:
    
        1. Expresso. Also completes preprocessing steps suggested in the ParaSpeechCaps repo
    
    Instructions:
    https://github.com/ajd12342/paraspeechcaps?tab=readme-ov-file#2-paraspeechcaps-dataset 
    """
    
    from datasets import load_dataset, load_dataset_builder
    import os
    
    data_dir = Path(path)
    annot_dir = data_dir / "annotations"
    # Cloning annotations
    if annot_dir.exists():
        print(f"ParaSpeechCaps repo already exists at {annot_dir}")
   
        
    else:
        subprocess.run(
            [
                "git", "clone",
                "https://github.com/ajd12342/paraspeechcaps.git",
                str(annot_dir)
            ],
            check=True,
        )
    
    psc_dataset_name = "ajd12342/paraspeechcaps"
    print("Checking if ParaSpeechCaps already cached.")
    builder = load_dataset_builder(psc_dataset_name)
    # this is the directory HF would use for the processed dataset
    cache_dir = builder.cache_dir

    if os.path.exists(cache_dir):
        print("ParaSpeechCaps already cached. SKIPPING.")
    else:
        # Download and cache the dataset
        print("Not cached. Downloadign ParaSpeechCaps")
        _ = load_dataset("ajd12342/paraspeechcaps")

        
    # Audio datasets
    
    data_dir_audio = data_dir / "audio"
        
    # Download Expresso dataset
    print("Downloading audio from EXPRESSO dataset")
    expresso_url = "https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar"
    name = "expresso"
    download_tar_dataset(expresso_url, data_dir_audio, name, gz=False)
    
    # Clone & download EARS dataset
    ears_git = "https://github.com/facebookresearch/ears_dataset.git"
    ears_folder = data_dir_audio / "EARS"
    if ears_folder.exists():
        print(f"EARS repo already exists at {ears_folder}")
   
    else:
        print("Cloning EARS repo to get download scripts.")
        subprocess.run(
            [
                "git", "clone",
                ears_git,
                str(ears_folder)
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
        "--styletalk",
        action="store_true",
        help="Download the styletalk dataset",
    )
    
    parser.add_argument(
        "--paraspeechcaps",
        action="store_true",
        help="Download the paraspeechcaps dataset",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all dataset",
    )

    args = parser.parse_args()

    if not (args.libritts or args.styletalk or args.paraspeechcaps or args.all):
        parser.error("Please specify at least one dataset to download.")

    if args.libritts:
        clone_libritts_p("data/raw/libritts")

    elif args.styletalk:
        clone_styletalk("data/raw/styletalk")
        
    elif args.paraspeechcaps:
        clone_paraspeechcaps("data/raw/paraspeechcaps")
        
    elif args.all:
        clone_libritts_p("data/raw/libritts")
        clone_styletalk("data/raw/styletalk")
        clone_paraspeechcaps("data/raw/paraspeechcaps")


if __name__ == "__main__":
    main()
