"""FastAPI Backend Server providing clinical endpoints, radiologist reviews, and audit trails."""

import io
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image

from ..models.inference import MedicalImageInferenceEngine
from ..dicom.converter import convert_image_to_dicom, read_dicom_image
from ..orthanc.client import OrthancClient

app = FastAPI(
    title="AI-Assisted Medical Imaging and Diagnosis System",
    description="End-to-End Clinical Diagnostic Platform with MONAI Segmentation, Grad-CAM Explainability, and Orthanc DICOM Integration.",
    version="1.0.0",
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent audit log file
AUDIT_LOG_FILE = Path("data/audit_logs.json")
AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
if not AUDIT_LOG_FILE.exists():
    AUDIT_LOG_FILE.write_text("[]", encoding="utf-8")

# Model checkpoints
CLASSIFIER_WEIGHTS = Path("weights/resnet18_isic_best.pth")
SEGMENTER_WEIGHTS = Path("weights/unet_isic_best.pth")

# Initialize Inference Engine with weights if available
engine = MedicalImageInferenceEngine(
    classifier_path=CLASSIFIER_WEIGHTS if CLASSIFIER_WEIGHTS.exists() else None,
    segmenter_path=SEGMENTER_WEIGHTS if SEGMENTER_WEIGHTS.exists() else None,
    pretrained=True if not CLASSIFIER_WEIGHTS.exists() else False,
)
orthanc_client = OrthancClient()


class ReviewDecision(BaseModel):
    study_id: str
    decision: str  # "Approved" or "Rejected"
    radiologist_name: str
    notes: Optional[str] = ""
    override_prediction: Optional[str] = None


@app.get("/api/health")
def health_check():
    orthanc_alive = orthanc_client.check_health()
    return {
        "status": "healthy",
        "service": "AI-Assisted Medical Imaging Platform",
        "orthanc_connected": orthanc_alive,
        "device": str(engine.device),
        "timestamp": datetime.utcnow().isoformat(),
        "disclaimer": "FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY. NOT FOR PRIMARY CLINICAL DIAGNOSIS.",
    }


@app.post("/api/analyze")
async def analyze_scan(file: UploadFile = File(...)):
    """Upload an image (JPEG/PNG or DICOM) and return predictions, segmentation, and Grad-CAM explanation."""
    import logging
    logger = logging.getLogger("uvicorn.error")
    logger.info(f"Incoming /api/analyze request for file: {file.filename}")

    try:
        contents = await file.read()
        filename = (file.filename or "unknown.jpg").lower()

        # Check if DICOM file
        if filename.endswith(".dcm") or contents.startswith(b"DICM", 128):
            image, dcm_meta = read_dicom_image(contents)
        else:
            image = Image.open(io.BytesIO(contents))
            dcm_meta = {"patient_id": "ANON_" + filename[:6], "modality": "XC"}

        result = engine.analyze(image)
        result["metadata"] = dcm_meta
        result["study_id"] = f"STD_{int(time.time()*1000)}"
        return result
    except Exception as e:
        logger.error(f"Error during /api/analyze: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process scan: {str(e)}")


@app.post("/api/review")
def record_radiologist_review(review: ReviewDecision):
    """Record radiologist review decision (Approve/Reject) into persistent audit log."""
    if review.decision not in ["Approved", "Rejected"]:
        raise HTTPException(status_code=400, detail="Decision must be 'Approved' or 'Rejected'")

    try:
        current_logs = json.loads(AUDIT_LOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        current_logs = []

    audit_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "study_id": review.study_id,
        "decision": review.decision,
        "radiologist_name": review.radiologist_name,
        "notes": review.notes,
        "override_prediction": review.override_prediction,
    }
    current_logs.append(audit_entry)
    AUDIT_LOG_FILE.write_text(json.dumps(current_logs, indent=2), encoding="utf-8")

    return {"message": "Audit log successfully updated", "entry": audit_entry}


@app.get("/api/audit-logs")
def get_audit_logs():
    """Retrieve full history of clinician decisions and audit trails."""
    try:
        return json.loads(AUDIT_LOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


@app.get("/api/orthanc/studies")
def get_orthanc_studies():
    """Retrieve studies currently held on Orthanc DICOM server."""
    if not orthanc_client.check_health():
        return {"connected": False, "studies": [], "message": "Orthanc server is not reachable on localhost:8042."}

    try:
        studies = orthanc_client.list_studies()
        details = [orthanc_client.get_study_details(s) for s in studies[:10]]
        return {"connected": True, "count": len(studies), "studies": details}
    except Exception as e:
        return {"connected": False, "error": str(e)}


# Mount static frontend files for single-container deployment (Render, Docker, etc.)
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Frontend index.html not found"}

    @app.get("/app.js")
    def serve_frontend_app_js():
        js_file = FRONTEND_DIR / "app.js"
        if js_file.exists():
            return FileResponse(str(js_file), media_type="application/javascript")
        return {"message": "app.js not found"}
