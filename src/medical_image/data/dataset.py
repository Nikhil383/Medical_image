"""Data loading, validation, downloading, and dataset classes for ISIC medical imaging."""

import os
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class ISICDataset(Dataset):
    """PyTorch Dataset for paired medical images, binary masks, and diagnostic labels."""

    def __init__(
        self,
        dataframe: Union[pl.DataFrame, "pandas.DataFrame"],
        image_size: int = 256,
        augment: bool = False,
    ):
        if hasattr(dataframe, "to_dicts"):
            self.records = dataframe.to_dicts()
        elif hasattr(dataframe, "to_dict"):
            self.records = dataframe.to_dict(orient="records")
        else:
            self.records = list(dataframe)

        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.records[index]

        image = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = (np.asarray(mask, dtype=np.uint8) >= 128).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                image_np = np.fliplr(image_np).copy()
                mask_np = np.fliplr(mask_np).copy()
            if random.random() < 0.5:
                image_np = np.flipud(image_np).copy()
                mask_np = np.flipud(mask_np).copy()

        # [C, H, W]
        image_t = torch.from_numpy(image_np).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask_np[None, ...]).float()
        label_t = torch.tensor(int(row["label"]), dtype=torch.long)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_t = (image_t - mean) / std

        return image_t, mask_t, label_t


def build_isic_metadata(
    raw_images_dir: Path,
    raw_labels_csv: Path,
    output_dir: Path,
    seed: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Parse raw ISIC files, validate image-mask pairings, build metadata and stratified splits."""
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_raw = pl.read_csv(raw_labels_csv, has_header=False, infer_schema=False)
    id_col = None
    diag_col = None

    for col in labels_raw.columns:
        values = [str(v).strip() for v in labels_raw[col].drop_nulls().to_list()]
        if any(v.startswith("ISIC_") for v in values):
            id_col = col
        if any(v.lower() in {"benign", "malignant"} for v in values):
            diag_col = col

    if id_col is None or diag_col is None:
        raise ValueError("Could not automatically locate image-ID and diagnosis columns.")

    labels = (
        labels_raw.select(
            pl.col(id_col).cast(pl.String).str.strip_chars().alias("image_id"),
            pl.col(diag_col).cast(pl.String).str.strip_chars().str.to_lowercase().alias("diagnosis"),
        )
        .filter(
            pl.col("image_id").str.starts_with("ISIC_")
            & pl.col("diagnosis").is_in(["benign", "malignant"])
        )
        .unique(subset=["image_id"])
        .with_columns(
            pl.when(pl.col("diagnosis") == "malignant")
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("label")
        )
    )

    image_paths = sorted(raw_images_dir.rglob("ISIC_*.jpg"))
    mask_paths = sorted(raw_images_dir.rglob("ISIC_*_Segmentation.png"))

    image_map = {p.stem: str(p.resolve()) for p in image_paths}
    mask_map = {p.stem.replace("_Segmentation", ""): str(p.resolve()) for p in mask_paths}

    images_df = pl.DataFrame([{"image_id": k, "image_path": v} for k, v in image_map.items()])
    masks_df = pl.DataFrame([{"image_id": k, "mask_path": v} for k, v in mask_map.items()])

    metadata = (
        labels.join(images_df, on="image_id", how="inner")
        .join(masks_df, on="image_id", how="inner")
        .select("image_id", "image_path", "mask_path", "diagnosis", "label")
        .sort("image_id")
    )

    meta_pd = metadata.to_pandas()
    train_df, temp_df = train_test_split(
        meta_pd, test_size=0.30, stratify=meta_pd["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed
    )

    train_pl = pl.from_pandas(train_df)
    val_pl = pl.from_pandas(val_df)
    test_pl = pl.from_pandas(test_df)

    metadata.write_csv(output_dir / "metadata.csv")
    train_pl.write_csv(output_dir / "train_metadata.csv")
    val_pl.write_csv(output_dir / "val_metadata.csv")
    test_pl.write_csv(output_dir / "test_metadata.csv")

    return train_pl, val_pl, test_pl


def create_synthetic_demo_dataset(
    output_dir: Path,
    num_samples: int = 12,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Generates synthetic paired dermatology images & segmentation masks for testing and offline execution."""
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    meta_dir = output_dir / "metadata"
    for d in [images_dir, masks_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    records = []
    rng = np.random.default_rng(42)

    for i in range(num_samples):
        img_id = f"ISIC_{9000000 + i}"
        is_malignant = 1 if i % 3 == 0 else 0
        diag = "malignant" if is_malignant else "benign"

        # Create realistic skin texture & synthetic lesion
        h, w = 256, 256
        # Base skin tone (pinkish-beige)
        base_color = np.array([210, 160, 140], dtype=np.float32)
        noise = rng.normal(0, 10, (h, w, 3)).astype(np.float32)
        img = np.clip(base_color + noise, 0, 255).astype(np.uint8)

        # Lesion center and radius
        cy, cx = rng.integers(100, 156), rng.integers(100, 156)
        ry, rx = rng.integers(30, 60), rng.integers(30, 60)
        yy, xx = np.ogrid[:h, :w]
        dist = ((yy - cy) ** 2) / (ry ** 2) + ((xx - cx) ** 2) / (rx ** 2)
        lesion_mask = (dist <= 1.0).astype(np.uint8) * 255

        # Darken lesion area on image
        lesion_color = np.array([60, 35, 25] if is_malignant else [110, 70, 50], dtype=np.float32)
        for c in range(3):
            img[:, :, c] = np.where(dist <= 1.0, lesion_color[c] + rng.normal(0, 8, (h, w)), img[:, :, c])
        img = np.clip(img, 0, 255).astype(np.uint8)

        img_path = images_dir / f"{img_id}.jpg"
        mask_path = masks_dir / f"{img_id}_Segmentation.png"

        Image.fromarray(img).save(img_path)
        Image.fromarray(lesion_mask).save(mask_path)

        records.append({
            "image_id": img_id,
            "image_path": str(img_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "diagnosis": diag,
            "label": is_malignant,
        })

    df = pl.DataFrame(records)
    n = len(df)
    train_df = df.slice(0, int(n * 0.6))
    val_df = df.slice(int(n * 0.6), int(n * 0.2))
    test_df = df.slice(int(n * 0.8), n - int(n * 0.8))

    df.write_csv(meta_dir / "metadata.csv")
    train_df.write_csv(meta_dir / "train_metadata.csv")
    val_df.write_csv(meta_dir / "val_metadata.csv")
    test_df.write_csv(meta_dir / "test_metadata.csv")

    return train_df, val_df, test_df
