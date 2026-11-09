"""Orthanc DICOM Server integration client and automated inference ingestion worker."""

import io
import time
from typing import Dict, Any, List, Optional
import requests
from PIL import Image

from ..dicom.converter import read_dicom_image, convert_image_to_dicom
from ..models.inference import MedicalImageInferenceEngine


class OrthancClient:
    """HTTP Client for communicating with an Orthanc DICOM PACS server."""

    def __init__(self, base_url: str = "http://localhost:8042", auth: Optional[tuple] = ("orthanc", "orthanc")):
        self.base_url = base_url.rstrip("/")
        self.auth = auth

    def check_health(self) -> bool:
        """Check if Orthanc server is reachable."""
        try:
            r = requests.get(f"{self.base_url}/system", auth=self.auth, timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def get_system_info(self) -> Dict[str, Any]:
        """Get system statistics and status from Orthanc."""
        r = requests.get(f"{self.base_url}/system", auth=self.auth, timeout=5)
        r.raise_for_status()
        return r.json()

    def list_patients(self) -> List[str]:
        """List patient IDs stored in Orthanc."""
        r = requests.get(f"{self.base_url}/patients", auth=self.auth, timeout=5)
        r.raise_for_status()
        return r.json()

    def list_studies(self) -> List[str]:
        """List study IDs stored in Orthanc."""
        r = requests.get(f"{self.base_url}/studies", auth=self.auth, timeout=5)
        r.raise_for_status()
        return r.json()

    def get_study_details(self, study_id: str) -> Dict[str, Any]:
        """Retrieve detailed metadata for a specific study."""
        r = requests.get(f"{self.base_url}/studies/{study_id}", auth=self.auth, timeout=5)
        r.raise_for_status()
        return r.json()

    def upload_dicom_file(self, dicom_bytes: bytes) -> Dict[str, Any]:
        """Upload raw DICOM file bytes to Orthanc."""
        headers = {"Content-Type": "application/dicom"}
        r = requests.post(f"{self.base_url}/instances", data=dicom_bytes, headers=headers, auth=self.auth, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_instance_dicom(self, instance_id: str) -> bytes:
        """Download raw DICOM file for an instance."""
        r = requests.get(f"{self.base_url}/instances/{instance_id}/file", auth=self.auth, timeout=10)
        r.raise_for_status()
        return r.content


class OrthancWatcherService:
    """Continuously monitors Orthanc DICOM server for incoming studies, triggers inference, and stores audit records."""

    def __init__(self, orthanc_client: OrthancClient, engine: MedicalImageInferenceEngine):
        self.orthanc = orthanc_client
        self.engine = engine
        self.processed_instances = set()
        self.results_cache: Dict[str, Any] = {}

    def poll_once(self) -> List[Dict[str, Any]]:
        """Checks for new instances, pulls DICOM, processes through AI inference."""
        if not self.orthanc.check_health():
            return []

        new_results = []
        try:
            r = requests.get(f"{self.orthanc.base_url}/instances", auth=self.orthanc.auth, timeout=5)
            if r.status_code != 200:
                return []
            instances = r.json()

            for inst_id in instances:
                if inst_id in self.processed_instances:
                    continue

                dcm_bytes = self.orthanc.get_instance_dicom(inst_id)
                pil_img, meta = read_dicom_image(dcm_bytes)

                analysis = self.engine.analyze(pil_img)
                record = {
                    "instance_id": inst_id,
                    "metadata": meta,
                    "analysis": analysis,
                    "processed_at": time.time(),
                }
                self.results_cache[inst_id] = record
                self.processed_instances.add(inst_id)
                new_results.append(record)

        except Exception as e:
            print(f"[OrthancWatcher] Polling error: {e}")

        return new_results
