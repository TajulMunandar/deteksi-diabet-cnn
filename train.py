import os
import sys
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json
import io

# Redirect stdout ke UTF-8 untuk konsistensi encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# === KONFIGURASI ===
DATASET_PATH = "dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train_images")
LABELS_CSV = os.path.join("train.csv")
IMG_SIZE = 224
EPOCHS = 20

image_index = {}


def build_image_index():
    print("🔍 Membangun index gambar...")
    for filename in os.listdir(TRAIN_PATH):
        base, ext = os.path.splitext(filename)
        if ext.lower() in [".jpg", ".jpeg", ".png"]:
            image_index[base] = os.path.join(TRAIN_PATH, filename)
    print(f"✅ Index selesai: {len(image_index)} gambar ditemukan")


def find_image(img_id):
    return image_index.get(img_id)


# === LOAD DATA ===
def load_images(image_ids, labels):
    print("📂 Memuat gambar dari dataset...")
    sys.stdout.flush()

    images = []
    y = []
    total = len(image_ids)

    for i, (img_id, label) in enumerate(zip(image_ids, labels)):
        if i % 100 == 0:  # Progress setiap 100 gambar
            print(f"📊 Progress: {i}/{total} gambar diproses")
            sys.stdout.flush()

        path = find_image(img_id)
        if path:
            try:
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    images.append(img)
                    y.append(label)
            except Exception as e:
                print(f"⚠️ Error memuat {img_id}: {str(e)}")

    print(f"✅ Total gambar berhasil dimuat: {len(images)}")
    sys.stdout.flush()
    return np.array(images, dtype=np.float32), np.array(y)


def main():
    try:
        print("🚀 Memulai training Diabetes Retinopathy Detection")
        print("=" * 50)
        sys.stdout.flush()

        print("📖 Membaca data label...")
        sys.stdout.flush()
        df = pd.read_csv(LABELS_CSV)

        image_ids = df["id_code"].values
        labels = df["diagnosis"].values

        print(
            f"📊 Dataset info: {len(image_ids)} sampel dengan {len(np.unique(labels))} kelas"
        )
        sys.stdout.flush()

        build_image_index()
        X, y = load_images(image_ids, labels)

        print("🔄 Preprocessing data...")
        sys.stdout.flush()
        X = X / 255.0  # Normalisasi
        y_cat = to_categorical(y, num_classes=5)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_cat, test_size=0.2, random_state=42
        )

        print(f"📊 Data split: Train={len(X_train)}, Validation={len(X_val)}")
        sys.stdout.flush()

        # === MODEL CNN ===
        print("🏗️ Membangun model CNN...")
        sys.stdout.flush()
        model = Sequential(
            [
                Conv2D(
                    32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)
                ),
                MaxPooling2D(2, 2),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPooling2D(2, 2),
                Conv2D(128, (3, 3), activation="relu"),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(128, activation="relu"),
                Dropout(0.5),
                Dense(5, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        print("📋 Model Summary:")
        model.summary()
        sys.stdout.flush()

        # === TRAINING ===
        print("🎯 Memulai pelatihan model...")
        sys.stdout.flush()

        datagen = ImageDataGenerator(
            rotation_range=15, zoom_range=0.1, horizontal_flip=True
        )
        datagen.fit(X_train)

        print(f"⏳ Training untuk {EPOCHS} epoch...")
        sys.stdout.flush()

        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            verbose=1,
        )

        # === SIMPAN MODEL ===
        print("💾 Menyimpan model...")
        sys.stdout.flush()
        model.save("model_dr.h5")

        final_acc = history.history["accuracy"][-1]
        final_loss = history.history["loss"][-1]
        final_val_acc = history.history.get("val_accuracy", [0])[-1]

        print("✅ Training selesai!")
        print(f"📈 Akurasi Training: {final_acc:.4f}")
        print(f"📈 Akurasi Validasi: {final_val_acc:.4f}")
        print(f"📉 Loss: {final_loss:.4f}")
        sys.stdout.flush()

        # Output JSON untuk Flask
        result = {
            "accuracy": round(final_acc * 100, 2),
            "val_accuracy": round(final_val_acc * 100, 2),
            "loss": round(final_loss, 4),
            "model_path": "model_dr.h5",
            "status": "success",
        }

        print(json.dumps(result))
        sys.stdout.flush()

    except Exception as e:
        error_result = {
            "accuracy": 0.0,
            "loss": 0.0,
            "model_path": "N/A",
            "status": "error",
            "error": str(e),
        }
        print(f"❌ Error during training: {str(e)}")
        print(json.dumps(error_result))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
