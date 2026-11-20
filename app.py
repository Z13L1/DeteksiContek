"""
DeteksiContek - Langkah 1
Alur: browser (kamera) -> kirim frame ke Flask -> YOLO -> kirim balik kotak (JSON) -> digambar di browser
"""
import base64
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

# ---------- 1. Muat model ----------
MODEL_PATH = os.path.join("models", "best.pt")

if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    MODEL_NAME = "best.pt (model kamu)"
else:
    # Cadangan supaya alurnya tetap bisa dicoba sebelum best.pt ditaruh di folder models/
    print("[PERINGATAN] models/best.pt tidak ditemukan, memakai yolov8n.pt (deteksi objek umum).")
    model = YOLO("yolov8n.pt")
    MODEL_NAME = "yolov8n.pt (cadangan - bukan model contek)"

print("Kelas model:", model.names)


# ---------- 2. Fungsi bantu ----------
def decode_image(data_url):
    """Teks base64 dari browser -> array gambar OpenCV (BGR)."""
    if "," in data_url:
        data_url = data_url.split(",")[1]
    img_bytes = base64.b64decode(data_url)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------- 3. Route ----------
@app.route("/")
def index():
    return render_template("index.html", model_name=MODEL_NAME)


@app.route("/api/detect", methods=["POST"])
def detect():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"success": False, "message": "Frame tidak ditemukan"}), 400

    frame = decode_image(data["image"])
    if frame is None:
        return jsonify({"success": False, "message": "Gagal membaca frame"}), 400

    conf = float(data.get("conf", 0.5))

    # Inti AI: satu baris ini menjalankan YOLO pada frame
    result = model.predict(frame, conf=conf, imgsz=640, verbose=False)[0]

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        detections.append({
            "label": model.names[class_id],
            "confidence": round(float(box.conf[0]), 3),
            "box": [x1, y1, x2, y2],
        })

    return jsonify({"success": True, "detections": detections})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
