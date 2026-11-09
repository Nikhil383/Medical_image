"""Automated download helper for CheXpert and Medical Segmentation Decathlon (MSD).

Note on CheXpert:
Stanford CheXpert requires a free data agreement registration. Direct raw links require your
registered download URL / credentials from: https://stanfordmlgroup.github.io/competitions/chexpert/
Alternatively, the CheXpert v1.0 Small subset can be downloaded via Kaggle CLI.

Medical Segmentation Decathlon (MSD):
Hosts 10 3D volumetric biomedical segmentation tasks (Brain, Heart, Liver, Lung, Prostate, etc.)
hosted openly on AWS Open Data / Academic Torrents / Google Drive.
"""

import os
import sys
import tarfile
import zipfile
from pathlib import Path
import requests
from tqdm.auto import tqdm

DOWNLOAD_DIR = Path("dataset_downloaders/downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Medical Segmentation Decathlon tasks direct URLs (via AWS Open Data / Zenodo / Academic archives)
MSD_TASKS = {
    "Task01_BrainTumour": "https://msd-for-monai.s3-eu-west-1.amazonaws.com/Task01_BrainTumour.tar",
    "Task02_Heart": "https://msd-for-monai.s3-eu-west-1.amazonaws.com/Task02_Heart.tar",
    "Task04_Hippocampus": "https://msd-for-monai.s3-eu-west-1.amazonaws.com/Task04_Hippocampus.tar",
    "Task05_Prostate": "https://msd-for-monai.s3-eu-west-1.amazonaws.com/Task05_Prostate.tar",
    "Task06_Lung": "https://msd-for-monai.s3-eu-west-1.amazonaws.com/Task06_Lung.tar",
    "Task07_Pancreas": "https://msd-for-monai.s3-eu-west-1.amazonaws.com/Task07_Pancreas.tar",
    "Task08_HepaticVessel": "https://msd-for-monai.s3-eu-west-1.amazonaws.com/Task08_HepaticVessel.tar",
    "Task09_Spleen": "https://msd-for-monai.s3-eu-west-1.amazonaws.com/Task09_Spleen.tar",
    "Task10_Colon": "https://msd-for-monai.s3-eu-west-1.amazonaws.com/Task10_Colon.tar",
}

def stream_download(url: str, dest_file: Path, chunk_size: int = 1024 * 1024):
    """Download a file with streaming progress bar."""
    if dest_file.exists() and dest_file.stat().st_size > 0:
        print(f"[Exists] {dest_file.name} already downloaded ({dest_file.stat().st_size / (1024*1024):.1f} MB).")
        return dest_file

    print(f"[Downloading] {dest_file.name} from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, stream=True, headers=headers, timeout=(15, 300))
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    part_file = dest_file.with_suffix(dest_file.suffix + ".part")

    with open(part_file, "wb") as f, tqdm(
        total=total_size if total_size else None,
        unit="B",
        unit_scale=True,
        desc=dest_file.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    part_file.replace(dest_file)
    print(f"[Done] Saved to {dest_file}")
    return dest_file

def download_msd_task(task_name: str = "Task09_Spleen", extract: bool = False):
    """Download one of the Medical Segmentation Decathlon tasks."""
    if task_name not in MSD_TASKS:
        print(f"Unknown task: {task_name}. Available tasks: {list(MSD_TASKS.keys())}")
        return

    url = MSD_TASKS[task_name]
    target_path = DOWNLOAD_DIR / f"{task_name}.tar"
    downloaded_tar = stream_download(url, target_path)

    if extract and downloaded_tar and downloaded_tar.exists():
        extract_folder = DOWNLOAD_DIR / task_name
        print(f"Extracting {downloaded_tar.name} to {extract_folder}...")
        with tarfile.open(downloaded_tar, "r") as tar:
            tar.extractall(path=DOWNLOAD_DIR)
        print("Extraction complete.")

def print_chexpert_instructions():
    """Prints instructions and Kaggle CLI commands to download CheXpert dataset."""
    guide = \"\"\"
================================================================================
                    CHEXPERT DATASET DOWNLOAD INSTRUCTIONS
================================================================================
The CheXpert dataset requires acceptance of the Stanford ML Group research use agreement.

Option A: Via Official Stanford ML Portal
1. Register for free at: https://stanfordmlgroup.github.io/competitions/chexpert/
2. Check your email for direct download links (CheXpert-v1.0-small.zip ~11 GB, or full ~439 GB).
3. Place the downloaded ZIP or extracted folder into:
   dataset_downloaders/downloads/chexpert/

Option B: Via Kaggle CLI (CheXpert v1.0 Small ~11 GB)
If you have a Kaggle account (kaggle.json in ~/.kaggle/):
   pip install kaggle
   kaggle datasets download -d ashirwadsangwan/chexpert -p dataset_downloaders/downloads/chexpert --unzip
================================================================================
\"\"\"
    print(guide)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Medical Dataset Downloader (MSD & CheXpert)")
    parser.add_argument("--msd-task", type=str, default="Task09_Spleen", choices=list(MSD_TASKS.keys()), help="Which MSD task to download (Task09_Spleen is lightweight)")
    parser.add_argument("--extract", action="store_true", help="Extract archive after downloading")
    parser.add_argument("--chexpert-info", action="store_true", help="Show CheXpert download instructions")
    args = parser.parse_args()

    if args.chexpert_info:
        print_chexpert_instructions()
    else:
        print(f"--- Downloading MSD: {args.msd_task} ---")
        download_msd_task(args.msd_task, extract=args.extract)
        print_chexpert_instructions()
