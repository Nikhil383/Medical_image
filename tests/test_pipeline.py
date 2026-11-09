import sys
import pytest
from pathlib import Path
from PIL import Image
import numpy as np
import torch

from medical_image.data.dataset import create_synthetic_demo_dataset, ISICDataset
from medical_image.models.networks import get_classifier, get_monai_segmenter, GradCAM
from medical_image.models.inference import MedicalImageInferenceEngine
from medical_image.dicom.converter import convert_image_to_dicom, read_dicom_image


def test_synthetic_data_generation(tmp_path):
    train_df, val_df, test_df = create_synthetic_demo_dataset(tmp_path, num_samples=6)
    assert len(train_df) > 0
    assert len(test_df) > 0

    dataset = ISICDataset(train_df, image_size=128)
    img_t, mask_t, label_t = dataset[0]
    assert img_t.shape == (3, 128, 128)
    assert mask_t.shape == (1, 128, 128)
    assert label_t.item() in (0, 1)


def test_models_forward_and_gradcam():
    classifier = get_classifier(model_name="resnet18", num_classes=2, pretrained=False)
    segmenter = get_monai_segmenter(spatial_dims=2, in_channels=3, out_channels=1)

    dummy_input = torch.randn(2, 3, 128, 128)
    cls_out = classifier(dummy_input)
    assert cls_out.shape == (2, 2)

    seg_out = segmenter(dummy_input)
    assert seg_out.shape == (2, 1, 128, 128)

    grad_cam = GradCAM(classifier)
    cam = grad_cam.generate(dummy_input[:1], target_class=1)
    assert cam.shape == (128, 128)
    assert 0.0 <= cam.min() <= cam.max() <= 1.0


def test_dicom_conversion_and_reading(tmp_path):
    img_np = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    dcm_file = tmp_path / "test_scan.dcm"

    ds = convert_image_to_dicom(img_np, output_path=dcm_file, patient_id="TEST_001")
    assert dcm_file.exists()

    loaded_img, meta = read_dicom_image(dcm_file)
    assert loaded_img.size == (128, 128)
    assert meta["patient_id"] == "TEST_001"
    assert meta["modality"] == "XC"


def test_end_to_end_inference():
    engine = MedicalImageInferenceEngine()
    test_img = Image.new("RGB", (128, 128), color=(200, 150, 130))
    res = engine.analyze(test_img)

    assert res["prediction"] in ["benign", "malignant"]
    assert "confidence" in res
    assert "visualizations" in res
    assert res["visualizations"]["scan_image"].startswith("data:image/")
    assert res["visualizations"]["gradcam_overlay"].startswith("data:image/")
