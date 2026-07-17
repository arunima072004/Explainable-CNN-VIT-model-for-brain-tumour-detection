# Project Workflow — Explainable CNN-ViT Brain Tumor Detection

## Dataset

**Brain Tumor MRI Dataset** (Kaggle) with 4 classes:
- `glioma_tumor`, `meningioma_tumor`, `pituitary_tumor`, `no_tumor`
- Split into `Training/` and `Testing/` folders
- Images are grayscale MRI scans stored as JPG/PNG

---

## Preprocessing & Transforms

Two separate transform pipelines, both using ImageNet normalization (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`) because both models use ImageNet-pretrained backbones:

- **ViT transform** — resizes to `224×224` (ViT's required input size, patch grid = 14×14)
- **CNN transform** — resizes to `128×128` (smaller, faster, sufficient for cropped regions)

Training transforms add augmentation (horizontal flip, rotation, color jitter, affine) to reduce overfitting. Test transforms are clean — just resize, tensor, normalize.

---

## Stage 1 — Binary Detection (ViT)

**Model:** `ViTBinaryClassifier`
- Backbone: `vit_base_patch16_224` from `timm` — pretrained on ImageNet, `num_classes=0` (features only)
- Custom head: `Linear(768→512) → ReLU → Dropout(0.3) → Linear(512→2)`
- Output: 2 logits → softmax → probability of `[no_tumor, tumor]`

**Training:**
- Dataset: all 4 classes, labels = 0 (no_tumor) or 1 (everything else)
- 20 epochs, AdamW optimizer, lr=1e-4, CosineAnnealingLR scheduler
- CrossEntropyLoss
- Best checkpoint at epoch 11 — val accuracy **95.43%**

**Inference:**
- Forward pass → softmax → if `prob[tumor] >= 0.45` → tumor detected
- Confidence score = `prob[1]` (tumor probability)

**Attention extraction:**
- Forward hook registered on `blocks[-1].attn.attn_drop` (last transformer block's attention dropout layer)
- Captures raw attention weight matrix of shape `[B, heads, N, N]` where N = 197 tokens (1 CLS + 196 patches)
- Take CLS token's attention to all patches: `attn[:, :, 0, 1:]` → average over heads → shape `[B, 196]`
- Reshape to `[B, 14, 14]` — the spatial attention map

---

## Stage 2a — Localization (Attention-based Crop)

Using the 14×14 attention map from Stage 1:

1. Resize attention map from `14×14` to full image size using `cv2.resize`
2. Normalize to `[0, 255]`, apply binary threshold at `0.6 × 255`
3. `cv2.findContours` on the binary mask
4. Take the largest contour → `cv2.boundingRect` → get `(x, y, w, h)`
5. Add 10px margin, clamp to image bounds
6. Crop that region from the original image

This localizes the tumor without ever training on bounding box annotations.

---

## Stage 2b — Classification (ResNet-50)

**Model:** `CNNTumorClassifier`
- Backbone: `resnet50` from torchvision, pretrained on ImageNet
- Custom head replaces final FC: `Linear(2048→512) → ReLU → Dropout(0.5) → Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→3)`
- Output: 3 logits → softmax → probabilities for `[glioma, meningioma, pituitary]`

**Training:**
- Dataset: only 3 tumor classes (no_tumor excluded)
- 30 epochs, AdamW, lr=1e-3, CosineAnnealingLR
- Best checkpoint at epoch 22 — val accuracy **76.12%**

**Inference:**
- Input: cropped tumor region from Stage 2a
- Forward pass → softmax → argmax = predicted tumor type
- Returns all 3 class probabilities

---

## Stage 2c — Grad-CAM (Explainability)

**Library:** `pytorch-grad-cam`

**Process:**
1. Target layer: `cnn_model.backbone.layer4[-1]` — last residual block of ResNet-50
2. Forward pass through CNN on the cropped region
3. Backward pass for the predicted class using `ClassifierOutputTarget(pred_idx)`
4. Grad-CAM computes: gradient of class score w.r.t. each feature map in layer4 → global average pool gradients → use as weights → weighted sum of feature maps → ReLU
5. Result: grayscale heatmap `(cropped_h, cropped_w)` — values 0→1, higher = more influential for the classification

**Placing heatmap back on full image:**
- Create a zero canvas the size of the original image
- Paste the cropped heatmap at the bounding box coordinates → `heatmap_full`

---

## Post-processing (Mask & Overlay)

**Mask generation:**
1. Normalize heatmap to `[0, 1]`
2. Threshold at user-specified value (default 0.5, adjustable via UI slider)
3. Morphological close then open with 7×7 elliptical kernel — smooths mask edges

**Overlay generation:**
- Blend mask onto original: masked pixels = `(1−α) × original + α × color`, α=0.45, color=red `(255, 50, 50)`

**Outputs returned to frontend (all base64 encoded):**
- `original` — original MRI image
- `heatmap` — Grad-CAM overlaid on image via `show_cam_on_image`
- `mask` — binary mask as RGB
- `overlay` — red-tinted overlay on original
- `bbox` — `[x, y, w, h]` for canvas bounding box drawing

---

## Backend (Flask API)

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Receives image + threshold, runs full pipeline, returns JSON |
| `/report` | POST | Receives result JSON, generates and returns a PDF report |
| `/health` | GET | Returns device info (CPU/GPU) |

CORS enabled so the React frontend can call from a different port. Models are loaded once at startup and reused across requests.

---

## Frontend (React + Vite)

- **Upload page:** drag-and-drop or file picker, sends `multipart/form-data` to `/predict` via axios
- **Threshold slider:** controls mask threshold, passed as a form field alongside the image
- **Results page:** 4-view image toggle (Original / Grad-CAM / Mask / Overlay), confidence bars, tumor type badge, all-class probability breakdown, bounding box drawn on canvas scaled to rendered image dimensions
- **PDF download:** calls `/report` with result JSON, receives blob, triggers browser download
- Vite dev server proxies `/predict`, `/report`, `/health` to `localhost:5000`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model training | PyTorch, timm, torchvision |
| XAI | pytorch-grad-cam, ViT attention hooks |
| Image processing | OpenCV, NumPy |
| Backend | Flask, flask-cors |
| PDF generation | ReportLab |
| Frontend | React, Vite, Tailwind CSS, axios |
| Dev tooling | Python venv, npm |
