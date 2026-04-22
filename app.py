"""
app.py  –  Flask backend for Lung Disease Diagnosis System
Supports dual XAI: Grad-CAM + LIME
"""

import os, uuid
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from utils.inference import LungDiagnosisPredictor

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, "static", "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "results")
MODEL_PATH     = os.path.join(BASE_DIR, "models", "best_model.pth")
ALLOWED_EXT    = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["SECRET_KEY"]         = os.urandom(24)

predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
        predictor = LungDiagnosisPredictor(MODEL_PATH)
    return predictor

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def unique_filename(original):
    ext = original.rsplit(".", 1)[1].lower()
    return f"{uuid.uuid4().hex}.{ext}"


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if "xray" not in request.files:
        return jsonify({"error": "No file uploaded. Field name must be 'xray'."}), 400
    file = request.files["xray"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or missing file."}), 400

    fname        = unique_filename(secure_filename(file.filename))
    stem         = fname.rsplit(".", 1)[0]
    upload_path  = os.path.join(UPLOAD_FOLDER, fname)
    gradcam_name = f"gradcam_{stem}.png"
    lime_name    = f"lime_{stem}.png"
    file.save(upload_path)

    try:
        pred = get_predictor().predict(
            image_path        = upload_path,
            save_gradcam_path = os.path.join(RESULTS_FOLDER, gradcam_name),
            save_lime_path    = os.path.join(RESULTS_FOLDER, lime_name),
            lime_samples      = 500,
        )
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    p = get_predictor()
    enriched = []
    for item in pred["predictions"]:
        info = p.get_disease_info(item["label"])
        enriched.append({**item, **info})

    return jsonify({
        "status":        "success",
        "top_label":     pred["top_label"],
        "top_prob":      pred["top_prob"],
        "predictions":   enriched,
        "original_url":  f"/static/uploads/{fname}",
        "gradcam_url":   f"/static/results/{gradcam_name}",
        "lime_url":      f"/static/results/{lime_name}",
        "lime_r2":       pred["lime_r2"],
        "lime_num_segs": pred["lime_num_segs"],
        "xai_summary":   pred["xai_summary"],
        "inference_ms":  pred["inference_ms"],
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_ready": os.path.exists(MODEL_PATH)})


if __name__ == "__main__":
    print("=" * 55)
    print("  PneumoScan — Grad-CAM + LIME")
    print("  http://localhost:5000")
    print("=" * 55)
    try:
        get_predictor()
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
    app.run(debug=True, host="0.0.0.0", port=5000)
