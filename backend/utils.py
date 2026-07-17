"""
utils.py - Mirrors the exact pipeline from project.ipynb
"""

import os
import base64
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import TUMOR_CLASSES, VIT_IMG_SIZE

# ── Exact transforms from notebook Cell 4 ────────────────────────────────────

VIT_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CNN_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ATTENTION_THRESHOLD = 0.6
CROP_MARGIN = 10


# ── Image helpers ─────────────────────────────────────────────────────────────

def read_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def ndarray_to_base64(img_rgb: np.ndarray, fmt: str = '.jpg') -> str:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(fmt, img_bgr)
    if not success:
        raise ValueError("Image encoding failed")
    b64 = base64.b64encode(buf).decode('utf-8')
    mime = 'image/jpeg' if fmt == '.jpg' else 'image/png'
    return f"data:{mime};base64,{b64}"


# ── Stage 1: Binary detection (ViT) ──────────────────────────────────────────

def detect_tumor(image_rgb: np.ndarray, vit_model, device) -> dict:
    """
    Exact replica of notebook inference:
      - vit_test_transform → unsqueeze → model → softmax
      - label 0 = no_tumor, label 1 = tumor  (BrainMRIDataset)
    """
    tensor = VIT_TRANSFORM(image_rgb).unsqueeze(0).to(device)

    vit_model.eval()
    with torch.no_grad():
        output = vit_model(tensor)
        probs  = F.softmax(output, dim=1)
        pred   = output.argmax(dim=1).item()

    confidence = probs[0, 1].item()
    has_tumor  = confidence >= 0.45  # slightly below 0.5 to account for model's confidence calibration

    # Get attention map for localisation (same hook as notebook)
    attn_map = vit_model.get_attention_map(tensor)
    attn_np  = attn_map[0].cpu().numpy() if attn_map is not None else None

    return {
        'has_tumor':     has_tumor,
        'confidence':    confidence,
        'attention_map': attn_np,   # shape (14,14) or None
    }


# ── Stage 2: Localise & crop (exact notebook Cell 8) ─────────────────────────

def localize_and_crop(image_rgb: np.ndarray, attn_map: np.ndarray) -> tuple:
    """
    Direct port of notebook's localize_and_crop_tumor():
      1. Resize 14×14 attention map to image size
      2. cv2.threshold at ATTENTION_THRESHOLD * 255
      3. Find largest contour → bounding rect + CROP_MARGIN
    Returns (cropped_rgb, (x, y, w, h))
    """
    h, w = image_rgb.shape[:2]

    if attn_map is None:
        crop_size = min(h, w) // 2
        x0, y0 = w // 4, h // 4
        return image_rgb[y0:y0 + crop_size, x0:x0 + crop_size], \
               (x0, y0, crop_size, crop_size)

    # Resize attention map to full image size
    attn_resized = cv2.resize(attn_map, (w, h))

    # Threshold — same as notebook
    _, binary = cv2.threshold(
        (attn_resized * 255).astype(np.uint8),
        int(ATTENTION_THRESHOLD * 255),
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        crop_size = min(h, w) // 2
        x0, y0 = w // 4, h // 4
        return image_rgb[y0:y0 + crop_size, x0:x0 + crop_size], \
               (x0, y0, crop_size, crop_size)

    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)

    # Add margin, clamp to image bounds (exact notebook logic)
    x  = max(0, x - CROP_MARGIN)
    y  = max(0, y - CROP_MARGIN)
    bw = min(w - x, bw + 2 * CROP_MARGIN)
    bh = min(h - y, bh + 2 * CROP_MARGIN)

    cropped = image_rgb[y:y + bh, x:x + bw]
    return cropped, (x, y, bw, bh)


# ── Stage 3: Classify cropped region (CNN) ───────────────────────────────────

def classify_tumor(cropped_rgb: np.ndarray, cnn_model, device) -> dict:
    """
    Exact replica of notebook inference on cropped region:
      - cnn_test_transform (128×128) → unsqueeze → model → softmax
    """
    tensor = CNN_TRANSFORM(cropped_rgb).unsqueeze(0).to(device)

    cnn_model.eval()
    with torch.no_grad():
        output   = cnn_model(tensor)
        probs    = F.softmax(output, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())

    return {
        'tumor_type':      TUMOR_CLASSES[pred_idx],
        'type_confidence': float(probs[pred_idx]),
        'all_probs':       {cls: round(float(p), 4)
                            for cls, p in zip(TUMOR_CLASSES, probs)},
        'pred_idx':        pred_idx,
    }


# ── Grad-CAM on cropped region ────────────────────────────────────────────────

def generate_gradcam(cropped_rgb: np.ndarray, cnn_model,
                     target_class: int, device) -> np.ndarray:
    """Grad-CAM on the cropped tumor region, resized back to cropped dims."""
    tensor = CNN_TRANSFORM(cropped_rgb).unsqueeze(0).to(device)
    target_layers = [cnn_model.backbone.layer4[-1]]

    cam = GradCAM(model=cnn_model, target_layers=target_layers)
    grayscale_cam = cam(
        input_tensor=tensor,
        targets=[ClassifierOutputTarget(target_class)]
    )[0]

    return cv2.resize(grayscale_cam,
                      (cropped_rgb.shape[1], cropped_rgb.shape[0])).astype(np.float32)


# ── Masking ───────────────────────────────────────────────────────────────────

def generate_mask(heatmap: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Heatmap → binary mask:
      1. Normalise to [0,1]
      2. Threshold
      3. Morphological close + open
    """
    norm   = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    binary = (norm > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed,  cv2.MORPH_OPEN,  kernel)
    return opened


def generate_overlay(image_rgb: np.ndarray, mask: np.ndarray,
                     color=(255, 50, 50), alpha=0.45) -> np.ndarray:
    overlay   = image_rgb.copy().astype(np.float32)
    mask_bool = mask > 0
    for c, val in enumerate(color):
        overlay[:, :, c] = np.where(
            mask_bool,
            (1 - alpha) * overlay[:, :, c] + alpha * val,
            overlay[:, :, c]
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def generate_bbox_image(image_rgb: np.ndarray, bbox: tuple) -> np.ndarray:
    result = image_rgb.copy()
    x, y, w, h = bbox
    cv2.rectangle(result, (x, y), (x + w, y + h), (255, 50, 50), 2)
    return result


def heatmap_to_rgb(heatmap: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
    img_norm = image_rgb.astype(np.float32) / 255.0
    if heatmap.shape[:2] != image_rgb.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    return show_cam_on_image(img_norm, heatmap, use_rgb=True)


# ── Full pipeline (mirrors notebook Cell 9: predict_pipeline) ─────────────────

def run_pipeline(image_rgb: np.ndarray, vit_model, cnn_model, device,
                 mask_threshold: float = 0.5) -> dict:

    h, w = image_rgb.shape[:2]
    original_b64 = ndarray_to_base64(image_rgb)

    # Stage 1 — detection
    detection  = detect_tumor(image_rgb, vit_model, device)
    has_tumor  = detection['has_tumor']
    confidence = detection['confidence']

    if not has_tumor:
        return {
            'tumor_detected': False,
            'confidence':     round(confidence, 4),
            'tumor_type':     'N/A',
            'bbox':           None,
            'heatmap':        None,
            'mask':           None,
            'overlay':        None,
            'original':       original_b64,
        }

    # Stage 2 — localise using ViT attention (exact notebook method)
    cropped, bbox = localize_and_crop(image_rgb, detection['attention_map'])

    # Stage 3 — classify cropped region
    classification = classify_tumor(cropped, cnn_model, device)
    pred_idx       = classification['pred_idx']

    # Grad-CAM on cropped region
    heatmap_cropped = generate_gradcam(cropped, cnn_model, pred_idx, device)

    # Place cropped heatmap back on full-image canvas
    x, y, bw, bh = bbox
    heatmap_full = np.zeros((h, w), dtype=np.float32)
    heatmap_full[y:y + bh, x:x + bw] = cv2.resize(heatmap_cropped, (bw, bh))

    mask        = generate_mask(heatmap_full, threshold=mask_threshold)
    overlay     = generate_overlay(image_rgb, mask)
    bbox_img    = generate_bbox_image(image_rgb, bbox)
    heatmap_rgb = heatmap_to_rgb(heatmap_full, image_rgb)
    mask_rgb    = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    return {
        'tumor_detected':  True,
        'confidence':      round(confidence, 4),
        'tumor_type':      classification['tumor_type'],
        'type_confidence': round(classification['type_confidence'], 4),
        'all_probs':       classification['all_probs'],
        'bbox':            list(bbox),
        'heatmap':         ndarray_to_base64(heatmap_rgb),
        'mask':            ndarray_to_base64(mask_rgb),
        'overlay':         ndarray_to_base64(overlay),
        'original':        original_b64,
        'bbox_image':      ndarray_to_base64(bbox_img),
    }
