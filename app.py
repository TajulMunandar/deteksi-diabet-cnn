from flask import (
    Flask,
    request,
    render_template,
    redirect,
    Response,
    stream_with_context,
    session,
)
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import subprocess
import json
import sys


# === Konfigurasi ===
UPLOAD_FOLDER = "static/upload"
MODEL_PATH = "model_dr.h5"
IMG_SIZE = 224

# === Setup Flask ===
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.secret_key = "ini-rahasia"
# === Load Model ===
model = load_model(MODEL_PATH)
class_names = [
    "No DR (Normal / Tidak ada tanda-tanda retinopati diabetik)",
    "Mild (Ringan – Perubahan kecil pada pembuluh darah retina)",
    "Moderate (Sedang – Perubahan sedang pada retina, perlu perhatian medis)",
    "Severe (Parah – Banyak pembuluh darah bocor, risiko kehilangan penglihatan)",
    "Proliferative DR (Sangat Parah – Pertumbuhan pembuluh darah baru abnormal, berisiko kebutaan)",
]


# === Prediksi Gambar ===
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    predicted_class = np.argmax(pred)
    predicted_label = class_names[predicted_class]

    # Mapping ke binary
    if predicted_class == 0:
        binary_label = "Negative Diabetes"
    else:
        binary_label = "Positive Diabetes"

    return predicted_label, binary_label


# === Routes ===
@app.route("/", methods=["GET", "POST"])
def index():
    training_result = session.pop("training_result", None)
    training_log = session.pop("training_log", None)
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            pred_5class, pred_binary = predict_image(filepath)

            return render_template(
                "index.html",
                uploaded_image=file.filename,
                pred_5class=pred_5class,
                pred_binary=pred_binary,
            )

    return render_template(
        "index.html", training_result=training_result, training_log=training_log
    )


@app.route("/stream-train")
def stream_train():
    def generate():
        try:
            # Gunakan encoding yang lebih fleksibel dan error handling
            process = subprocess.Popen(
                [
                    sys.executable,
                    "train.py",
                ],  # Gunakan sys.executable untuk konsistensi
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",  # Ganti karakter bermasalah dengan ?
                bufsize=1,
                universal_newlines=True,
            )

            for line in process.stdout:
                yield f"data: {line.strip()}\n\n"

            process.stdout.close()
            return_code = process.wait()

            if return_code == 0:
                yield f"data: ✅ Training selesai dengan sukses.\n\n"
            else:
                yield f"data: ❌ Training selesai dengan error (kode: {return_code}).\n\n"

        except Exception as e:
            yield f"data: ❌ Error during training: {str(e)}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/train", methods=["POST"])
def train_model():
    try:
        # Gunakan encoding yang konsisten dan error handling
        process = subprocess.run(
            [sys.executable, "train.py"],  # Gunakan sys.executable
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",  # Ganti karakter bermasalah dengan ?
            timeout=3600,  # Timeout 1 jam untuk training
        )

        output = process.stdout
        error_output = process.stderr

        # Cari hasil JSON di akhir output
        try:
            # Ambil beberapa baris terakhir untuk mencari JSON
            lines = output.strip().split("\n")
            result_json = None

            # Cari dari belakang untuk menemukan JSON yang valid
            for line in reversed(lines):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        result_json = line
                        break
                    except:
                        continue

            if result_json:
                training_result = json.loads(result_json)
            else:
                raise ValueError("Tidak ditemukan hasil JSON yang valid")

        except Exception as e:
            training_result = {
                "accuracy": 0.0,
                "loss": 0.0,
                "model_path": "N/A",
                "error": f"Parse error: {str(e)}",
            }

        # Gabungkan output dan error jika ada
        full_log = output
        if error_output:
            full_log += "\n--- STDERR ---\n" + error_output

        # Simpan output log
        session["training_log"] = full_log
        session["training_result"] = training_result

    except subprocess.TimeoutExpired:
        session["training_log"] = "❌ Training timeout (lebih dari 1 jam)"
        session["training_result"] = {
            "accuracy": 0.0,
            "loss": 0.0,
            "model_path": "N/A",
            "error": "Training timeout",
        }
    except Exception as e:
        session["training_log"] = f"❌ Error menjalankan training: {str(e)}"
        session["training_result"] = {
            "accuracy": 0.0,
            "loss": 0.0,
            "model_path": "N/A",
            "error": str(e),
        }

    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
