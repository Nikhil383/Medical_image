Project name-"AI-Assisted Medical Imaging and Diagnosis System"

goal=A platform where X-ray, CT, or MRI scans are uploaded, analyzed by deep learning models, shown with explanation heatmaps, and reviewed by a doctor."

- use UV instead of pip(i have created project directory and integrated with uv)
- use the the file in notebook/AI_Assisted_Medical_Imaging_ISIC.ipynb to develop end to end capstone project.

Step-by-Step Instructions

1. Choose public data (CheXpert or NIH ChestX-ray14 for classification, Medical Segmentation Decathlon for segmentation). Learn the DICOM format.(i choose ISIC datset for segmentation and classifcation i was not getting any dataset)(do not mention this in my project,its for you to work in ISIC dataset only)

2. Preprocess: resize, normalize, augment. Version your data with DVC.

3. Train a classification model with transfer learning (DenseNet or EfficientNet) and a segmentation model (3D U-Net in MONAI).

4. Add Grad-CAM so each prediction shows which region influenced it.

5. Install an Orthanc DICOM server, then write a service that pulls new studies, runs inference, and stores results.

6. Build a React viewer showing the scan, the AI overlay, a confidence score, and Approve/Reject buttons for the radiologist. Keep an audit log of every decision.

7. Containerize with Docker and document the model's limits and bias. Include a clear 'for education only, not clinical use' notice.

- Required Tech Stacks: MONAI, PyTorch, Orthanc DICOM server, Grad-CAM, DVC, Docker, React


Deliverables Checklist
- Working demo (deployed or screen-recorded)
- GitHub repository with clean commits, README, and setup instructions
- Architecture diagram and data or API documentation
- Test results and, where relevant, performance or accuracy numbers

FOR YOU
- if you have nay doubt then ask rather than hallucniating