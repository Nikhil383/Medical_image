"""Automated DVC pipeline definition."""

from pathlib import Path
import yaml

dvc_pipeline = {
    "stages": {
        "prepare_data": {
            "cmd": "uv run python -c \"from medical_image.data.dataset import create_synthetic_demo_dataset; from pathlib import Path; create_synthetic_demo_dataset(Path('data/demo'), num_samples=30)\"",
            "deps": ["src/medical_image/data/dataset.py"],
            "outs": ["data/demo/metadata/metadata.csv"],
        },
        "generate_dicom": {
            "cmd": "uv run python -c \"from medical_image.dicom.converter import convert_image_to_dicom; from pathlib import Path; [convert_image_to_dicom(p, Path('data/sample_dicom') / f'{p.stem}.dcm', patient_id=f'PT_{i+1:03d}') for i, p in enumerate(list(Path('data/demo/images').glob('*.jpg'))[:10])]\"",
            "deps": ["data/demo/metadata/metadata.csv", "src/medical_image/dicom/converter.py"],
            "outs": ["data/sample_dicom"],
        },
    }
}

Path("dvc.yaml").write_text(yaml.dump(dvc_pipeline, sort_keys=False), encoding="utf-8")
print("dvc.yaml generated successfully!")
