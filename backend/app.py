"""
app.py - Flask API for Brain Tumor Detection & Classification
"""

import os
import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from model import load_models
from utils import read_image_from_bytes, run_pipeline
from report import generate_pdf

app = Flask(__name__)
CORS(app)  # Allow requests from React frontend

# ── Device & model paths ──────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

VIT_WEIGHTS = os.environ.get('VIT_WEIGHTS', 'vit_binary_best.pth')
CNN_WEIGHTS = os.environ.get('CNN_WEIGHTS', 'cnn_tumor_type_best.pth')

# Load models once at startup
vit_model, cnn_model = None, None

def get_models():
    global vit_model, cnn_model
    if vit_model is None or cnn_model is None:
        if not os.path.exists(VIT_WEIGHTS) or not os.path.exists(CNN_WEIGHTS):
            print("[WARN] Weight files not found — initializing with random weights for demo.")
            _save_dummy_weights()
        vit_model, cnn_model = load_models(VIT_WEIGHTS, CNN_WEIGHTS, device)
        print("[INFO] Models loaded successfully.")
    return vit_model, cnn_model


def _save_dummy_weights():
    """Save randomly-initialized model weights so the pipeline runs without trained files."""
    from model import ViTBinaryClassifier, CNNTumorClassifier
    if not os.path.exists(VIT_WEIGHTS):
        m = ViTBinaryClassifier(pretrained=False)
        torch.save(m.state_dict(), VIT_WEIGHTS)
        print(f"[INFO] Saved dummy ViT weights → {VIT_WEIGHTS}")
    if not os.path.exists(CNN_WEIGHTS):
        m = CNNTumorClassifier(num_classes=3, pretrained=False)
        torch.save(m.state_dict(), CNN_WEIGHTS)
        print(f"[INFO] Saved dummy CNN weights → {CNN_WEIGHTS}")
    # Write a marker so utils.py knows to use demo mode
    open('.demo_mode', 'w').close()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(device)})


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Form-data: image (file), threshold (optional float, default 0.5)

    Returns JSON with detection result, classification, and base64 images.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided. Use key "image".'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    # Optional masking threshold from form data
    try:
        threshold = float(request.form.get('threshold', 0.5))
        threshold = max(0.1, min(0.9, threshold))  # clamp to [0.1, 0.9]
    except ValueError:
        threshold = 0.5

    try:
        vit, cnn = get_models()
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 503

    try:
        image_bytes = file.read()
        image_rgb = read_image_from_bytes(image_bytes)
    except Exception as e:
        return jsonify({'error': f'Failed to read image: {e}'}), 400

    try:
        result = run_pipeline(image_rgb, vit, cnn, device, mask_threshold=threshold)
    except Exception as e:
        return jsonify({'error': f'Inference failed: {e}'}), 500

    return jsonify(result)


@app.route('/report', methods=['POST'])
def report():
    """
    POST /report  (JSON body)
    Accepts the result dict already computed by /predict and generates a PDF.
    Body: { result: {...}, filename: "scan.jpg" }
    """
    data = request.get_json(silent=True)
    if not data or 'result' not in data:
        return jsonify({'error': 'Missing result data.'}), 400

    filename = data.get('filename', 'scan.jpg')
    result   = data['result']

    try:
        pdf_bytes = generate_pdf(result, filename=filename)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'PDF generation failed: {e}'}), 500

    import io
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'brain_tumor_report_{filename.rsplit(".", 1)[0]}.pdf'
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
