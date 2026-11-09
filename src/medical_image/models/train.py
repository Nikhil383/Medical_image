"""Training workflows for Classification and MONAI Segmentation."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .networks import get_classifier, get_monai_segmenter
from ..data.dataset import ISICDataset


def train_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train classification model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_classifier(model_name="resnet18", num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, _, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_y, all_probs = [], []
        with torch.no_grad():
            for images, _, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_y.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
        val_loss /= len(val_loader.dataset)

        preds = (np.array(all_probs) >= 0.5).astype(int)
        acc = float(accuracy_score(all_y, preds)) if all_y else 0.0

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": acc})
        print(f"[Classifier] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}")

        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)

    return {"history": history, "best_val_loss": best_val_loss}


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    denominator = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def train_segmenter(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train MONAI UNet model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_monai_segmenter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            bce = F.binary_cross_entropy_with_logits(logits, masks)
            dice = dice_loss(logits, masks)
            loss = 0.5 * bce + 0.5 * dice
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        dice_scores = []
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                bce = F.binary_cross_entropy_with_logits(logits, masks)
                dice = dice_loss(logits, masks)
                val_loss += (0.5 * bce + 0.5 * dice).item() * images.size(0)
                dice_scores.append(1.0 - dice.item())
        val_loss /= len(val_loader.dataset)
        avg_dice = float(np.mean(dice_scores)) if dice_scores else 0.0

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_dice": avg_dice})
        print(f"[Segmenter] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {avg_dice:.4f}")

        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)

    return {"history": history, "best_val_loss": best_val_loss}
