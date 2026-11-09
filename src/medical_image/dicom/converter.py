"""DICOM utilities for medical image conversion, metadata tagging, and extraction."""

import io
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
from datetime import datetime
import numpy as np
from PIL import Image

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, SecondaryCaptureImageStorage


def convert_image_to_dicom(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    patient_id: str = "PATIENT_001",
    patient_name: str = "Anonymous^Patient",
    study_description: str = "Dermatology Scan Evaluation",
    modality: str = "XC",  # External Camera / Dermatoscopy
) -> pydicom.Dataset:
    """Convert a standard RGB/Grayscale image to a standard DICOM FileDataset."""
    if isinstance(image_input, (str, Path)):
        im = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        im = Image.fromarray(image_input)
    else:
        im = image_input

    # Ensure RGB
    im = im.convert("RGB")
    np_img = np.asarray(im, dtype=np.uint8)

    # Setup DICOM File Meta Information
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    sop_instance_uid = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    filename = str(output_path) if output_path else "temp.dcm"
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Patient & Study attributes
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.PatientBirthDate = "19800101"
    ds.PatientSex = "O"

    now = datetime.now()
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.StudyID = "1"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = sop_instance_uid
    ds.SOPClassUID = SecondaryCaptureImageStorage

    ds.Modality = modality
    ds.SeriesDescription = "Dermoscopic Inspection"
    ds.StudyDescription = study_description

    # Image Pixel Module attributes
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.Rows = np_img.shape[0]
    ds.Columns = np_img.shape[1]
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = np_img.tobytes()

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(output_path, write_like_original=False)

    return ds


def read_dicom_image(dicom_path_or_bytes: Union[str, Path, bytes, io.BytesIO]) -> Tuple[Image.Image, Dict[str, Any]]:
    """Extract PIL Image and patient/study metadata from a DICOM object."""
    if isinstance(dicom_path_or_bytes, bytes):
        dcm = pydicom.dcmread(io.BytesIO(dicom_path_or_bytes))
    elif isinstance(dicom_path_or_bytes, io.BytesIO):
        dcm = pydicom.dcmread(dicom_path_or_bytes)
    else:
        dcm = pydicom.dcmread(str(dicom_path_or_bytes))

    pixel_arr = dcm.pixel_array

    # Handle grayscale vs color
    if pixel_arr.ndim == 2:
        # Grayscale
        im = Image.fromarray(pixel_arr).convert("RGB")
    elif pixel_arr.ndim == 3:
        im = Image.fromarray(pixel_arr.astype(np.uint8), mode="RGB")
    else:
        raise ValueError(f"Unsupported DICOM array dimensions: {pixel_arr.shape}")

    metadata = {
        "patient_id": str(getattr(dcm, "PatientID", "UNKNOWN")),
        "patient_name": str(getattr(dcm, "PatientName", "ANONYMOUS")),
        "modality": str(getattr(dcm, "Modality", "UNKNOWN")),
        "study_description": str(getattr(dcm, "StudyDescription", "")),
        "study_instance_uid": str(getattr(dcm, "StudyInstanceUID", "")),
        "series_instance_uid": str(getattr(dcm, "SeriesInstanceUID", "")),
        "sop_instance_uid": str(getattr(dcm, "SOPInstanceUID", "")),
    }

    return im, metadata
