# AI-Assisted Medical Imaging and Clinical Diagnosis System

**Project Type:** End-to-End Applied AI / Computer Vision Capstone  
**Author:** AI/ML Engineering Intern  
**Tech Stack:** PyTorch, MONAI, Grad-CAM, Orthanc DICOM PACS, FastAPI, React, Docker, DVC, UV  

---

> ⚠️ **MANDATORY CLINICAL & REGULATORY DISCLAIMER**  
> **FOR INVESTIGATIONAL, RESEARCH, AND EDUCATIONAL USE ONLY.**  
> This platform and its deep learning models are strictly decision-support prototypes and are **not** FDA/CE-cleared diagnostic medical devices. Outputs must never replace professional clinical evaluation, patient triage, or direct histological biopsy. All AI predictions must be validated by a licensed physician.

---

## 1. Real-World Problem Statement

In modern clinical diagnostic pathways, dermatologists and radiologists face two critical challenges:
1. **Severe Workload & Diagnostic Latency:** Over 5 million skin cancer screenings and millions of diagnostic imaging studies occur annually. High clinical volume leads to clinician fatigue and diagnostic delays. Early detection of malignant melanoma is critical, where 5-year survival drops from >99% in localized stages to under 35% in metastatic stages.
2. **The "Black Box" Barrier in Clinical AI:** Standard deep learning classifiers output probability scores without spatial or visual reasoning. Without knowing *why* an algorithm flagged an image (e.g., did it look at the actual lesion boundary or a distracting surgical pen mark, hair, or ruler artifact?), doctors cannot responsibly trust or adopt AI predictions into patient care.
3. **PACS & Hospital Workflow Isolation:** Most AI prototypes operate in isolation on standard JPEGs instead of integrating with native hospital standards: **DICOM** (Digital Imaging and Communications in Medicine) and **PACS** (Picture Archiving and Communication Systems) with clinician-in-the-loop audit logs.

### The Solution
An end-to-end clinical AI platform where:
- Medical scans (DICOM or dermoscopy images) are ingested.
- **Deep Learning Classifiers** (DenseNet / ResNet) predict malignancy risk with confidence scores.
- **MONAI U-Net Models** delineate anatomical lesion boundaries.
- **Grad-CAM Saliency Maps** provide visual explanations of the neural network's focus.
- Hospital **Orthanc PACS** integration automates study ingestion.
- A **React Radiologist Dashboard** allows the clinician to inspect overlays, record clinical remarks, and explicitly **Approve or Reject** predictions with an immutable audit log.

---

## 2. Key Learnings & Engineering Takeaways

As an AI/ML intern building this project end-to-end, here are the core skills and engineering lessons acquired:

### A. Medical Data Engineering & Preprocessing
- **DICOM Protocol Standards:** Mastered parsing and writing DICOM files with `pydicom`, handling SOP Class UIDs, Patient ID anonymization, pixel representations, and photometric interpretations (RGB vs Monochromatic).
- **Geometric Invariance in Paired Augmentation:** Learned how medical segmentation masks must undergo identical spatial transformations (synchronized horizontal/vertical flips, affine transforms) as raw scans to prevent label misalignment.
- **Data Versioning with DVC:** Implemented reproducible data tracking and pipeline staging (`dvc.yaml`) to decouple dataset storage from Git commits.

### B. Deep Learning & Computer Vision
- **Transfer Learning on Imbalanced Clinical Data:** Fine-tuned convolutional backbones (ResNet18 / DenseNet121) on clinical lesion data using weighted cross-entropy and AdamW optimization.
- **Specialized Medical Segmentation with MONAI:** Built a U-Net architecture trained with a hybrid objective (**Dice Loss + Binary Cross-Entropy**) to handle foreground-background class imbalance where lesions occupy a small fraction of the image frame.
- **Explainable AI (Grad-CAM):** Extracted gradient activations from the final bottleneck convolutional layer to compute class activation heatmaps, exposing whether the model focused on true biological margins or peripheral artifacts.

### C. System Architecture & Full-Stack Deployment
- **Clinician-in-the-Loop Human Review:** Designed an interactive workflow where AI acts solely as an assistant. Every clinical action (Approve/Reject) requires doctor identification and is permanently recorded into an audit trail.
- **High-Performance Python Environment with `uv`:** Used Astral `uv` for 10x faster deterministic dependency resolution and virtual environment isolation.
- **PACS Integration & Container Orchestration:** Configured an open-source Orthanc DICOM PACS server and orchestrated the entire microservices architecture (Inference API + PACS + Web Viewer) using Docker Compose.

---

## 3. System Architecture & Workflow

```
                                      +---------------------------------------------+
                                      |            Radiologist Web Viewer           |
                                      |            (React 18 / Tailwind)            |
                                      +----------------------+----------------------+
                                                             |
                                           REST API (Upload) |  Approve / Reject Decision
                                                             v
+--------------------------+          +---------------------------------------------+
|   Orthanc DICOM Server   | <------> |          FastAPI Inference Engine           |
|     (PACS Port 8042)     |  DICOM   |              (PyTorch & MONAI)              |
+--------------------------+          +----------------------+----------------------+
                                                             |
                                      +----------------------+----------------------+
                                      |                      |                      |
                                      v                      v                      v
                             +-----------------+   +------------------+   +-------------------+
                             |  Classification |   | MONAI UNet Model |   | Grad-CAM Saliency |
                             | (Benign/Malign) |   |  (Lesion Masks)  |   |   (Explanation)   |
                             +-----------------+   +------------------+   +-------------------+
```

---

## 4. Project Directory Structure

```
├── data/
│   ├── demo/                  # Synthetic dermatology images & segmentation masks
│   ├── sample_dicom/          # Standard DICOM files (.dcm) with patient metadata
│   └── audit_logs.json        # Permanent clinician sign-off audit trail
├── frontend/
│   ├── index.html             # React viewer host application
│   └── app.js                 # Complete React dashboard, viewer canvas, audit table
├── notebook/
│   └── AI_Assisted_Medical_Imaging_ISIC.ipynb  # Baseline experimental training notebook
├── src/
│   └── medical_image/
│       ├── api/               # FastAPI endpoints (/api/analyze, /api/review, /api/health)
│       ├── data/              # Dataset loading, ISIC ingestion, validation, splitting
│       ├── dicom/             # DICOM conversion & metadata extraction
│       ├── models/            # Classification, MONAI U-Net, Grad-CAM, Inference engine
│       └── orthanc/           # Orthanc PACS REST client & watcher service
├── tests/
│   └── test_pipeline.py       # Automated unit and integration test suite
├── docker-compose.yml         # Multi-container orchestration (Orthanc, API, UI)
├── Dockerfile                 # Production API container image
├── dvc.yaml                   # DVC reproducible pipeline definition
├── pyproject.toml             # Project dependency specification
└── README.md                  # Comprehensive project documentation
```

---

## 5. Step-by-Step Execution Guide

### Prerequisites
- Python 3.12 (installed and managed via `uv`)
- Web browser (Chrome / Edge / Firefox)
- Docker & Docker Compose *(optional, for full containerized stack)*

---

### Step 1: Clone Repository & Setup Environment

```bash
# Clone the repository
git clone <your-repository-url>
cd medical_image

# Create virtual environment with uv
uv venv --python 3.12 .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies (CPU PyTorch + MONAI + FastAPI)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -e .
```

---

### Step 2: Run Automated Test Suite
Verify that data loaders, MONAI models, Grad-CAM, DICOM converters, and inference run cleanly:
```bash
uv run pytest tests/test_pipeline.py -v
```

---

### Step 3: Launch the Services (Local Demo)

Open two terminal windows:

#### Terminal 1 — Start the FastAPI AI Inference Engine:
```bash
uv run uvicorn medical_image.api.server:app --reload --port 8000
```
- API Health Status: `http://localhost:8000/api/health`
- Interactive Swagger API Documentation: `http://localhost:8000/docs`

#### Terminal 2 — Start the React Radiologist Dashboard:
```bash
python -m http.server 3000 --directory frontend
```
Now navigate to **`http://localhost:3000`** in your browser.

---

### Step 4: Test the End-to-End Clinical Review Workflow

1. Open **`http://localhost:3000`**.
2. Click **"Select Scan"** and choose a sample file:
   - Sample image: `data/demo/images/ISIC_9000000.jpg` **OR**
   - Sample DICOM: `data/sample_dicom/ISIC_9000000.dcm`.
3. Click **"Analyze Scan"**:
   - The AI will return the **prediction** (`benign` or `malignant`) and confidence percentage.
   - The **MONAI U-Net** identifies the lesion region and computes lesion area coverage.
   - The interactive canvas displays the **Grad-CAM attention heatmap** overlaid on the lesion.
   - Use the toggle buttons (**Raw Scan**, **Grad-CAM Overlay**, **MONAI Mask**) to inspect features.
4. Enter your clinician name (e.g., `Dr. Alex Smith, MD`) and diagnostic notes.
5. Click **Approve** or **Reject**:
   - Notice the review is permanently appended to the **Regulatory Audit Trail** table at the bottom and saved to `data/audit_logs.json`.

---

### Step 5: (Optional) Full Stack via Docker Compose
To spin up the entire system including the **Orthanc DICOM Server (PACS)**:
```bash
docker-compose up --build
```
- **React Radiologist Viewer:** `http://localhost:3000`
- **AI Inference Engine:** `http://localhost:8000`
- **Orthanc DICOM Server Explorer:** `http://localhost:8042`

---

## 6. Model Limitations & Demographic Bias Analysis

1. **Demographic & Skin-Tone Bias:** The underlying ISIC public dermatology archive is predominantly sampled from populations with Fitzpatrick skin phototypes I to III (light skin tones). As an ML intern, it was crucial to document that performance can suffer degradation on darker skin tones (phototypes IV–VI) due to historical training representation gaps.
2. **Clinical Artifact Sensitivity:** Convolutions are sensitive to dermoscopy artifacts such as air bubbles, dermoscopic gel glare, surgical marker lines, and hair follicles. Incorporating **Grad-CAM** serves as a vital safeguard: if the heatmap activates on a surgical ink mark rather than the pigment lesion, the doctor can instantly catch and reject the erroneous prediction.
3. **Domain & Modality Transfer:** A model trained on dermoscopic epiluminescence microscopy cannot be naively transferred to standard mobile phone photographs or histology slides without domain adaptation and calibration.

---

## 7. Deliverables & Evaluation Checklist

- [x] **Working Interactive Demo:** React radiologist dashboard with real-time multi-view canvas (Raw scan / Grad-CAM / MONAI mask).
- [x] **Deep Learning & Explainability:** MONAI U-Net segmentation + PyTorch Transfer Learning classification + Grad-CAM heatmap generation.
- [x] **Hospital Imaging Standards:** DICOM reading & writing via `pydicom`, secondary capture metadata tagging, and Orthanc PACS REST API client.
- [x] **Clinician-in-the-Loop:** Approve / Reject sign-off buttons and persistent regulatory audit trail (`data/audit_logs.json`).
- [x] **Automated Tests:** 100% passing test suite (`tests/test_pipeline.py`).
- [x] **Containerization:** Production `Dockerfile` and `docker-compose.yml`.
- [x] **Package Management:** Fast, modern Python setup utilizing `uv`.
- [x] **Clinical Notice:** Prominent educational disclaimer and algorithmic bias documentation.
