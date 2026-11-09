"""ISIC Dataset Downloader."""

import os
import requests
from pathlib import Path
from tqdm.auto import tqdm

ISIC_2016_FILES = {
    "training_data.zip": (
        "https://isic-challenge-data.s3.amazonaws.com/2016/"
        "ISBI2016_ISIC_Part3B_Training_Data.zip"
    ),
    "training_labels.csv": (
        "https://isic-challenge-data.s3.amazonaws.com/2016/"
        "ISBI2016_ISIC_Part3B_Training_GroundTruth.csv"
    ),
}

def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024):
    """Download a file with streaming progress bar."""
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already present: {destination.name} ({destination.stat().st_size:,} bytes)")
        return

    print(f"Downloading {destination.name} ...")
    response = requests.get(url, stream=True, timeout=(30, 180))
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    temp_path = destination.with_suffix(destination.suffix + ".part")

    with open(temp_path, "wb") as f, tqdm(
        total=total if total else None,
        unit="B",
        unit_scale=True,
        desc=destination.name,
    ) as progress:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                progress.update(len(chunk))

    temp_path.replace(destination)
    print(f"Saved: {destination}")

def download_isic_data(raw_dir: Path):
    """Download ISIC challenge dataset archives."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in ISIC_2016_FILES.items():
        download_file(url, raw_dir / filename)
